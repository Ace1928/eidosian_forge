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
class INotifyWrapper:
    """
    Abstract class wrapping access to inotify's functions. This is an
    internal class.
    """

    @staticmethod
    def create():
        """
        Factory method instanciating and returning the right wrapper.
        """
        if ctypes:
            inotify = _CtypesLibcINotifyWrapper()
            if inotify.init():
                return inotify
        if inotify_syscalls:
            inotify = _INotifySyscallsWrapper()
            if inotify.init():
                return inotify

    def get_errno(self):
        """
        Return None is no errno code is available.
        """
        return self._get_errno()

    def str_errno(self):
        code = self.get_errno()
        if code is None:
            return 'Errno: no errno support'
        return 'Errno=%s (%s)' % (os.strerror(code), errno.errorcode[code])

    def inotify_init(self):
        return self._inotify_init()

    def inotify_add_watch(self, fd, pathname, mask):
        assert isinstance(pathname, str)
        return self._inotify_add_watch(fd, pathname, mask)

    def inotify_rm_watch(self, fd, wd):
        return self._inotify_rm_watch(fd, wd)