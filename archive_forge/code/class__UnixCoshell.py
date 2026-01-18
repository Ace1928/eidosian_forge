from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import locale
import os
import re
import signal
import subprocess
from googlecloudsdk.core.util import encoding
import six
class _UnixCoshell(_UnixCoshellBase):
    """The unix local coshell implementation.

  This implementation preserves coshell process state across Run().

  Attributes:
    _status_fd: The read side of the pipe where the coshell write 1 char status
      lines. The status line is used to mark the exit of the currently running
      command.
  """
    SHELL_PATH = '/bin/bash'

    def __init__(self, stdout=1, stderr=2):
        super(_UnixCoshell, self).__init__()
        try:
            caller_shell_status_fd = os.dup(self.SHELL_STATUS_FD)
        except OSError:
            caller_shell_status_fd = -1
        os.dup2(1, self.SHELL_STATUS_FD)
        try:
            caller_shell_stdin_fd = os.dup(self.SHELL_STDIN_FD)
        except OSError:
            caller_shell_stdin_fd = -1
        os.dup2(0, self.SHELL_STDIN_FD)
        self._status_fd, w = os.pipe()
        os.dup2(w, self.SHELL_STATUS_FD)
        os.close(w)
        coshell_command_line = encoding.GetEncodedValue(os.environ, COSHELL_ENV)
        if coshell_command_line:
            shell_command = coshell_command_line.split(' ')
        else:
            shell_command = [self.SHELL_PATH]
        additional_kwargs = {} if six.PY2 else {'restore_signals': False}
        self._shell = subprocess.Popen(shell_command, env=os.environ, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr, close_fds=False, **additional_kwargs)
        if caller_shell_status_fd >= 0:
            os.dup2(caller_shell_status_fd, self.SHELL_STATUS_FD)
            os.close(caller_shell_status_fd)
        else:
            os.close(self.SHELL_STATUS_FD)
        if caller_shell_stdin_fd >= 0:
            os.dup2(caller_shell_stdin_fd, self.SHELL_STDIN_FD)
            os.close(caller_shell_stdin_fd)
        else:
            os.close(self.SHELL_STDIN_FD)
        self._GetUserConfigDefaults()

    def Close(self):
        """Closes the coshell connection and release any resources."""
        if self._status_fd >= 0:
            os.close(self._status_fd)
            self._status_fd = -1
        try:
            self._WriteLine('exit')
        except (IOError, ValueError):
            pass
        return self._ShellStatus(self._shell.returncode)

    def _Run(self, command, check_modes=True):
        """Runs command in the coshell and waits for it to complete."""
        self._SendCommand('command eval {command} <&{fdin} && echo 0 >&{fdstatus} || {{ status=$?; echo $status 1>&{fdstatus}; _status $status; }}'.format(command=self._Quote(command), fdstatus=self.SHELL_STATUS_FD, fdin=self.SHELL_STDIN_FD))
        status = self._GetStatus()
        if check_modes:
            if re.search('\\bset\\s+[-+]o\\s+\\w', command):
                self._GetModes()
            if re.search('\\bcd\\b', command):
                self.GetPwd()
        return status

    def Communicate(self, args, quote=True):
        """Runs args and returns the list of output lines, up to first empty one.

    Args:
      args: The list of command line arguments.
      quote: Shell quote args if True.

    Returns:
      The list of output lines from command args up to the first empty line.
    """
        if quote:
            command = ' '.join([self._Quote(arg) for arg in args])
        else:
            command = ' '.join(args)
        self._SendCommand('{command} >&{fdstatus}\n'.format(command=command, fdstatus=self.SHELL_STATUS_FD))
        lines = []
        line = []
        while True:
            try:
                c = self._ReadStatusChar()
            except (IOError, OSError, ValueError):
                self._Exited()
            if c in (None, b'\n'):
                if not line:
                    break
                lines.append(self._Decode(b''.join(line).rstrip()))
                line = []
            else:
                line.append(c)
        return lines