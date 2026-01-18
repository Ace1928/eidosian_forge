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
def __add_watch(self, path, mask, proc_fun, auto_add, exclude_filter):
    """
        Add a watch on path, build a Watch object and insert it in the
        watch manager dictionary. Return the wd value.
        """
    path = self.__format_path(path)
    if auto_add and (not mask & IN_CREATE):
        mask |= IN_CREATE
    wd = self._inotify_wrapper.inotify_add_watch(self._fd, path, mask)
    if wd < 0:
        return wd
    watch = Watch(wd=wd, path=path, mask=mask, proc_fun=proc_fun, auto_add=auto_add, exclude_filter=exclude_filter)
    self._wmd[wd] = watch
    log.debug('New %s', watch)
    return wd