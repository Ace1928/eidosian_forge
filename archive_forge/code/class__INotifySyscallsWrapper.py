import sys
import threading
import os
import select
import struct
import fcntl
import errno
import termios
import array
import logging
import atexit
from collections import deque
from datetime import datetime, timedelta
import time
import re
import asyncore
import glob
import locale
import subprocess
class _INotifySyscallsWrapper(INotifyWrapper):

    def __init__(self):
        self._last_errno = None

    def init(self):
        assert inotify_syscalls
        return True

    def _get_errno(self):
        return self._last_errno

    def _inotify_init(self):
        try:
            fd = inotify_syscalls.inotify_init()
        except IOError as err:
            self._last_errno = err.errno
            return -1
        return fd

    def _inotify_add_watch(self, fd, pathname, mask):
        try:
            wd = inotify_syscalls.inotify_add_watch(fd, pathname, mask)
        except IOError as err:
            self._last_errno = err.errno
            return -1
        return wd

    def _inotify_rm_watch(self, fd, wd):
        try:
            ret = inotify_syscalls.inotify_rm_watch(fd, wd)
        except IOError as err:
            self._last_errno = err.errno
            return -1
        return ret