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
class _MinGWCoshell(_UnixCoshellBase):
    """The MinGW local coshell implementation.

  This implementation preserves coshell process state across Run().

  NOTE: The Windows subprocess module passes fds 0,1,2 to the child process and
  no others. It is possble to pass handles that can be converted to/from fds,
  but the child process needs to know what handles to convert back to fds. Until
  we figure out how to reconstitute handles as fds >= 3 we are stuck with
  restricting fds 0,1,2 to be /dev/tty, via shell redirection, for Run(). For
  internal communication fds 0,1 are pipes. Luckily this works for the shell
  interactive prompt. Unfortunately this fails for the test environment.
  """
    SHELL_PATH = None
    STDIN_PATH = '/dev/tty'
    STDOUT_PATH = '/dev/tty'

    def __init__(self):
        super(_MinGWCoshell, self).__init__()
        self._shell = self._Popen()
        self._GetUserConfigDefaults()

    def _Popen(self):
        """Mockable popen+startupinfo so we can test on Unix."""
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dWflags = subprocess.CREATE_NEW_PROCESS_GROUP
        return subprocess.Popen([self.SHELL_PATH], stdin=subprocess.PIPE, stdout=subprocess.PIPE, startupinfo=startupinfo)

    def Close(self):
        """Closes the coshell connection and release any resources."""
        try:
            self._WriteLine('exit')
        except (IOError, ValueError):
            pass
        return self._ShellStatus(self._shell.returncode)

    def _GetStatus(self):
        """Gets the status of the last command sent to the coshell."""
        status_string = self._ReadLine()
        if status_string.endswith(self.SHELL_STATUS_EXIT):
            c = self.SHELL_STATUS_EXIT
            status_string = status_string[:-1]
        else:
            c = ''
        if not status_string.isdigit() or c == self.SHELL_STATUS_EXIT:
            self._Exited()
        return int(status_string)

    def _Run(self, command, check_modes=True):
        """Runs command in the coshell and waits for it to complete."""
        self._SendCommand("command eval {command} <'{stdin}' >>'{stdout}' && echo 0 || {{ status=$?; echo 1; (exit $status); }}".format(command=self._Quote(command), stdin=self.STDIN_PATH, stdout=self.STDOUT_PATH))
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
        self._SendCommand(command + '\n')
        lines = []
        while True:
            try:
                line = self._ReadLine()
            except (IOError, OSError, ValueError):
                self._Exited()
            if not line:
                break
            lines.append(line)
        return lines

    def Interrupt(self):
        """Sends the interrupt signal to the coshell."""
        self._shell.send_signal(signal.CTRL_C_EVENT)