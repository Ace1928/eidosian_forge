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
def _get_py_exe():

    def attempt(exe):
        try:
            subprocess.check_call([exe, '-V'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError:
            return None
        else:
            return exe
    env = os.environ.copy()
    base = getattr(sys, '_base_executable', None)
    if WINDOWS and sys.version_info >= (3, 7) and (base is not None):
        env['__PYVENV_LAUNCHER__'] = sys.executable
        return (base, env)
    elif GITHUB_ACTIONS:
        return (sys.executable, env)
    elif MACOS:
        exe = attempt(sys.executable) or attempt(os.path.realpath(sys.executable)) or attempt(which('python%s.%s' % sys.version_info[:2])) or attempt(psutil.Process().exe())
        if not exe:
            raise ValueError("can't find python exe real abspath")
        return (exe, env)
    else:
        exe = os.path.realpath(sys.executable)
        assert os.path.exists(exe), exe
        return (exe, env)