from __future__ import annotations
import os
import select
import shlex
import signal
import subprocess
import sys
from typing import ClassVar, Mapping
import param
from pyviz_comms import JupyterComm
from ..io.callbacks import PeriodicCallback
from ..util import edit_readonly, lazy_load
from .base import Widget
class TerminalSubprocess(param.Parameterized):
    """
    The TerminalSubProcess is a utility class that makes running
    subprocesses via the Terminal easy.
    """
    args = param.ClassSelector(class_=(str, list), doc='\n        The arguments used to run the subprocess. This may be a string\n        or a list. The string cannot contain spaces. See subprocess.run\n        docs for more details.')
    kill = param.Action(doc='Kills the running process', constant=True)
    kwargs = param.Dict(doc='\n        Any other arguments to run the subprocess. See subprocess.run\n        docs for more details.')
    running = param.Boolean(default=False, constant=True, doc='\n        Whether or not the subprocess is running.')
    _child_pid = param.Integer(default=0, doc='Child process id')
    _fd = param.Integer(default=0, doc='Child file descriptor.')
    _max_read_bytes = param.Integer(default=1024 * 20)
    _periodic_callback = param.ClassSelector(class_=PeriodicCallback, doc='\n        Watches the subprocess for output')
    _period = param.Integer(default=50, doc='Period length of _periodic_callback')
    _terminal = param.Parameter(constant=True, allow_refs=False, doc='\n        The Terminal to which the subprocess is connected.')
    _timeout_sec = param.Integer(default=0)
    _watcher = param.Parameter(doc='Watches the subprocess for user input')

    def __init__(self, terminal, **kwargs):
        super().__init__(_terminal=terminal, kill=self._kill, **kwargs)

    @staticmethod
    def _quote(command):
        return ''.join([shlex.quote(c) for c in command])

    def _clean_args(self, args):
        if isinstance(args, str):
            return self._quote(args)
        if isinstance(args, list):
            return [self._quote(arg) for arg in args]
        return args

    def run(self, *args, **kwargs):
        """
        Runs a subprocess command.
        """
        import pty
        if not args:
            args = self.args
        if not args:
            raise ValueError('Error. No args provided')
        if self.running:
            raise ValueError('Error. A child process is already running. Cannot start another.')
        args = self._clean_args(args)
        if self.kwargs:
            kwargs = {**self.kwargs, **kwargs}
        child_pid, fd = pty.fork()
        if child_pid == 0:
            try:
                result = subprocess.run(args, **kwargs)
                print(str(result))
            except FileNotFoundError as e:
                print(str(e) + "\nCompletedProcess('FileNotFoundError')")
        else:
            self._child_pid = child_pid
            self._fd = fd
            self._set_winsize()
            self._periodic_callback = PeriodicCallback(callback=self._forward_subprocess_output_to_terminal, period=self._period)
            self._periodic_callback.start()
            self._watcher = self._terminal.param.watch(self._forward_terminal_input_to_subprocess, 'value', onlychanged=False)
            with param.edit_constant(self):
                self.running = True

    @param.depends('_terminal.ncols', '_terminal.nrows', watch=True)
    def _set_winsize(self):
        if self._fd is None or not self._terminal.nrows or (not self._terminal.ncols):
            return
        import fcntl
        import struct
        import termios
        winsize = struct.pack('HHHH', self._terminal.nrows, self._terminal.ncols, 0, 0)
        try:
            fcntl.ioctl(self._fd, termios.TIOCSWINSZ, winsize)
        except OSError:
            pass

    def _kill(self, *events):
        child_pid = self._child_pid
        self._reset()
        if child_pid:
            os.killpg(os.getpgid(child_pid), signal.SIGTERM)
            self._terminal.write(f'\nThe process {child_pid} was killed\n')
        else:
            self._terminal.write('\nNo running process to kill\n')

    def _reset(self):
        self._fd = 0
        self._child_pid = 0
        if self._periodic_callback:
            self._periodic_callback.stop()
            self._periodic_callback = None
        if self._watcher:
            self._terminal.param.unwatch(self._watcher)
        with param.edit_constant(self):
            self.running = False

    @staticmethod
    def _remove_last_line_from_string(value):
        return value[:value.rfind('CompletedProcess')]

    def _decode_utf8_on_boundary(self, fd, max_read_bytes, max_extra_bytes=2):
        """UTF-8 characters can be multi-byte so need to decode on correct boundary"""
        data = os.read(fd, max_read_bytes)
        for _ in range(max_extra_bytes + 1):
            try:
                return data.decode('utf-8')
            except UnicodeDecodeError:
                data = data + os.read(fd, 1)
        raise UnicodeError('Could not find decode boundary for UTF-8')

    def _forward_subprocess_output_to_terminal(self):
        if not self._fd:
            return
        data_ready, _, _ = select.select([self._fd], [], [], self._timeout_sec)
        if not data_ready:
            return
        output = self._decode_utf8_on_boundary(self._fd, self._max_read_bytes)
        if 'CompletedProcess' in output:
            self._reset()
            output = self._remove_last_line_from_string(output)
        self._terminal.write(output)

    def _forward_terminal_input_to_subprocess(self, *events):
        if self._fd:
            os.write(self._fd, self._terminal.value.encode())

    @param.depends('args', watch=True)
    def _validate_args(self):
        args = self.args
        if isinstance(args, str) and ' ' in args:
            raise ValueError(f"The args '{args}' provided contains spaces. They must instead be provided as the\n                list {args.split(' ')}")

    @param.depends('_period', watch=True)
    def _update_periodic_callback(self):
        if self._periodic_callback:
            self._periodic_callback.period = self._period

    def __repr__(self, depth=None):
        return f'TerminalSubprocess(args={self.args}, running={self.running})'