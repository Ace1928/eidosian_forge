from __future__ import print_function
import atexit
import contextlib
import ctypes
import errno
import functools
import gc
import inspect
import os
import platform
import random
import re
import select
import shlex
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
import warnings
from socket import AF_INET
from socket import AF_INET6
from socket import SOCK_STREAM
import psutil
from psutil import AIX
from psutil import LINUX
from psutil import MACOS
from psutil import NETBSD
from psutil import OPENBSD
from psutil import POSIX
from psutil import SUNOS
from psutil import WINDOWS
from psutil._common import bytes2human
from psutil._common import debug
from psutil._common import memoize
from psutil._common import print_color
from psutil._common import supports_ipv6
from psutil._compat import PY3
from psutil._compat import FileExistsError
from psutil._compat import FileNotFoundError
from psutil._compat import range
from psutil._compat import super
from psutil._compat import u
from psutil._compat import unicode
from psutil._compat import which
class ThreadTask(threading.Thread):
    """A thread task which does nothing expect staying alive."""

    def __init__(self):
        super().__init__()
        self._running = False
        self._interval = 0.001
        self._flag = threading.Event()

    def __repr__(self):
        name = self.__class__.__name__
        return '<%s running=%s at %#x>' % (name, self._running, id(self))

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args, **kwargs):
        self.stop()

    def start(self):
        """Start thread and keep it running until an explicit
        stop() request. Polls for shutdown every 'timeout' seconds.
        """
        if self._running:
            raise ValueError('already started')
        threading.Thread.start(self)
        self._flag.wait()

    def run(self):
        self._running = True
        self._flag.set()
        while self._running:
            time.sleep(self._interval)

    def stop(self):
        """Stop thread execution and and waits until it is stopped."""
        if not self._running:
            raise ValueError('already stopped')
        self._running = False
        self.join()