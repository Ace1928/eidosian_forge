import os
import sys
import time
import pty
import tty
import errno
import signal
from contextlib import contextmanager
import ptyprocess
from ptyprocess.ptyprocess import use_native_pty_fork
from .exceptions import ExceptionPexpect, EOF, TIMEOUT
from .spawnbase import SpawnBase
from .utils import (
def __interact_copy(self, escape_character=None, input_filter=None, output_filter=None):
    """This is used by the interact() method.
        """
    while self.isalive():
        if self.use_poll:
            r = poll_ignore_interrupts([self.child_fd, self.STDIN_FILENO])
        else:
            r, w, e = select_ignore_interrupts([self.child_fd, self.STDIN_FILENO], [], [])
        if self.child_fd in r:
            try:
                data = self.__interact_read(self.child_fd)
            except OSError as err:
                if err.args[0] == errno.EIO:
                    break
                raise
            if data == b'':
                break
            if output_filter:
                data = output_filter(data)
            self._log(data, 'read')
            os.write(self.STDOUT_FILENO, data)
        if self.STDIN_FILENO in r:
            data = self.__interact_read(self.STDIN_FILENO)
            if input_filter:
                data = input_filter(data)
            i = -1
            if escape_character is not None:
                i = data.rfind(escape_character)
            if i != -1:
                data = data[:i]
                if data:
                    self._log(data, 'send')
                self.__interact_writen(self.child_fd, data)
                break
            self._log(data, 'send')
            self.__interact_writen(self.child_fd, data)