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
def _check_fds(self, fun):
    """Makes sure num_fds() (POSIX) or num_handles() (Windows) does
        not increase after calling a function.  Used to discover forgotten
        close(2) and CloseHandle syscalls.
        """
    before = self._get_num_fds()
    self.call(fun)
    after = self._get_num_fds()
    diff = after - before
    if diff < 0:
        raise self.fail('negative diff %r (gc probably collected a resource from a previous test)' % diff)
    if diff > 0:
        type_ = 'fd' if POSIX else 'handle'
        if diff > 1:
            type_ += 's'
        msg = '%s unclosed %s after calling %r' % (diff, type_, fun)
        raise self.fail(msg)