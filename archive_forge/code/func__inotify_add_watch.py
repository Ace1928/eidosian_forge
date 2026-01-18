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
def _inotify_add_watch(self, fd, pathname, mask):
    assert self._libc is not None
    pathname = pathname.encode(sys.getfilesystemencoding())
    pathname = ctypes.create_string_buffer(pathname)
    return self._libc.inotify_add_watch(fd, pathname, mask)