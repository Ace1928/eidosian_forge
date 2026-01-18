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
def _which(program, paths=None):
    """takes a program name or full path, plus an optional collection of search
    paths, and returns the full path of the requested executable.  if paths is
    specified, it is the entire list of search paths, and the PATH env is not
    used at all.  otherwise, PATH env is used to look for the program"""

    def is_exe(file_path):
        return os.path.exists(file_path) and os.access(file_path, os.X_OK) and os.path.isfile(os.path.realpath(file_path))
    found_path = None
    fpath, fname = os.path.split(program)
    if fpath:
        program = canonicalize(program)
        if is_exe(program):
            found_path = program
    else:
        paths_to_search = []
        if isinstance(paths, (tuple, list)):
            paths_to_search.extend(paths)
        else:
            env_paths = os.environ.get('PATH', '').split(os.pathsep)
            paths_to_search.extend(env_paths)
        for path in paths_to_search:
            exe_file = os.path.join(canonicalize(path), program)
            if is_exe(exe_file):
                found_path = exe_file
                break
    return found_path