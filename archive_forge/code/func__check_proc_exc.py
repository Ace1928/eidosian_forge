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
def _check_proc_exc(self, proc, exc):
    self.assertIsInstance(exc, psutil.Error)
    self.assertEqual(exc.pid, proc.pid)
    self.assertEqual(exc.name, proc._name)
    if exc.name:
        self.assertNotEqual(exc.name, '')
    if isinstance(exc, psutil.ZombieProcess):
        self.assertEqual(exc.ppid, proc._ppid)
        if exc.ppid is not None:
            self.assertGreaterEqual(exc.ppid, 0)
    str(exc)
    repr(exc)