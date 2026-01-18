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
class SessionContent(object):

    def __init__(self):
        self.chars = deque(maxlen=50000)
        self.lines = deque(maxlen=5000)
        self.line_chars = []
        self.last_line = ''
        self.cur_char = ''

    def append_char(self, char):
        if char == '\n':
            line = self.cur_line
            self.last_line = line
            self.lines.append(line)
            self.line_chars = []
        else:
            self.line_chars.append(char)
        self.chars.append(char)
        self.cur_char = char

    @property
    def cur_line(self):
        line = ''.join(self.line_chars)
        return line