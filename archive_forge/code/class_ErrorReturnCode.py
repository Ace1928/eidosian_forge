import asyncio
from collections import deque
import errno
import fcntl
import gc
import getpass
import glob as glob_module
import inspect
import logging
import os
import platform
import pty
import pwd
import re
import select
import signal
import stat
import struct
import sys
import termios
import textwrap
import threading
import time
import traceback
import tty
import warnings
import weakref
from asyncio import Queue as AQueue
from contextlib import contextmanager
from functools import partial
from importlib import metadata
from io import BytesIO, StringIO, UnsupportedOperation
from io import open as fdopen
from locale import getpreferredencoding
from queue import Empty, Queue
from shlex import quote as shlex_quote
from types import GeneratorType, ModuleType
from typing import Any, Dict, Type, Union
class ErrorReturnCode(Exception):
    __metaclass__ = ErrorReturnCodeMeta
    ' base class for all exceptions as a result of a command\'s exit status\n    being deemed an error.  this base class is dynamically subclassed into\n    derived classes with the format: ErrorReturnCode_NNN where NNN is the exit\n    code number.  the reason for this is it reduces boiler plate code when\n    testing error return codes:\n\n        try:\n            some_cmd()\n        except ErrorReturnCode_12:\n            print("couldn\'t do X")\n\n    vs:\n        try:\n            some_cmd()\n        except ErrorReturnCode as e:\n            if e.exit_code == 12:\n                print("couldn\'t do X")\n\n    it\'s not much of a savings, but i believe it makes the code easier to read '
    truncate_cap = 750

    def __reduce__(self):
        return (self.__class__, (self.full_cmd, self.stdout, self.stderr, self.truncate))

    def __init__(self, full_cmd, stdout, stderr, truncate=True):
        self.exit_code = self.exit_code
        self.full_cmd = full_cmd
        self.stdout = stdout
        self.stderr = stderr
        self.truncate = truncate
        exc_stdout = self.stdout
        if truncate:
            exc_stdout = exc_stdout[:self.truncate_cap]
            out_delta = len(self.stdout) - len(exc_stdout)
            if out_delta:
                exc_stdout += f'... ({out_delta} more, please see e.stdout)'.encode()
        exc_stderr = self.stderr
        if truncate:
            exc_stderr = exc_stderr[:self.truncate_cap]
            err_delta = len(self.stderr) - len(exc_stderr)
            if err_delta:
                exc_stderr += f'... ({err_delta} more, please see e.stderr)'.encode()
        msg = f'\n\n  RAN: {self.full_cmd}\n\n  STDOUT:\n{exc_stdout.decode(DEFAULT_ENCODING, 'replace')}\n\n  STDERR:\n{exc_stderr.decode(DEFAULT_ENCODING, 'replace')}'
        super(ErrorReturnCode, self).__init__(msg)