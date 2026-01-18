import itertools
import logging
import os
import queue
import re
import signal
import struct
import sys
import threading
import time
from collections import defaultdict
import wandb
class _WindowSizeChangeHandler:

    def __init__(self):
        self._fds = set()

    def _register(self):
        old_handler = signal.signal(signal.SIGWINCH, lambda *_: None)

        def handler(signum, frame):
            if callable(old_handler):
                old_handler(signum, frame)
            self.handle_window_size_change()
        signal.signal(signal.SIGWINCH, handler)
        self._old_handler = old_handler

    def _unregister(self):
        signal.signal(signal.SIGWINCH, self._old_handler)

    def add_fd(self, fd):
        if not self._fds:
            self._register()
        self._fds.add(fd)
        self.handle_window_size_change()

    def remove_fd(self, fd):
        if fd in self._fds:
            self._fds.remove(fd)
            if not self._fds:
                self._unregister()

    def handle_window_size_change(self):
        try:
            win_size = fcntl.ioctl(0, termios.TIOCGWINSZ, '\x00' * 8)
            rows, cols, xpix, ypix = struct.unpack('HHHH', win_size)
        except OSError:
            return
        if cols == 0:
            return
        win_size = struct.pack('HHHH', rows, cols, xpix, ypix)
        for fd in self._fds:
            fcntl.ioctl(fd, termios.TIOCSWINSZ, win_size)