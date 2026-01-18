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
class StreamBufferer(object):
    """this is used for feeding in chunks of stdout/stderr, and breaking it up
    into chunks that will actually be put into the internal buffers.  for
    example, if you have two processes, one being piped to the other, and you
    want that, first process to feed lines of data (instead of the chunks
    however they come in), OProc will use an instance of this class to chop up
    the data and feed it as lines to be sent down the pipe"""

    def __init__(self, buffer_type, encoding=DEFAULT_ENCODING, decode_errors='strict'):
        self.type = buffer_type
        self.buffer = []
        self.n_buffer_count = 0
        self.encoding = encoding
        self.decode_errors = decode_errors
        self._use_up_buffer_first = False
        self._buffering_lock = threading.RLock()
        self.log = Logger('stream_bufferer')

    def change_buffering(self, new_type):
        self.log.debug('acquiring buffering lock for changing buffering')
        self._buffering_lock.acquire()
        self.log.debug('got buffering lock for changing buffering')
        try:
            if new_type == 0:
                self._use_up_buffer_first = True
            self.type = new_type
        finally:
            self._buffering_lock.release()
            self.log.debug('released buffering lock for changing buffering')

    def process(self, chunk):
        self.log.debug('acquiring buffering lock to process chunk (buffering: %d)', self.type)
        self._buffering_lock.acquire()
        self.log.debug('got buffering lock to process chunk (buffering: %d)', self.type)
        try:
            if self.type == 0:
                if self._use_up_buffer_first:
                    self._use_up_buffer_first = False
                    to_write = self.buffer
                    self.buffer = []
                    to_write.append(chunk)
                    return to_write
                return [chunk]
            elif self.type == 1:
                total_to_write = []
                nl = '\n'.encode(self.encoding)
                while True:
                    newline = chunk.find(nl)
                    if newline == -1:
                        break
                    chunk_to_write = chunk[:newline + 1]
                    if self.buffer:
                        chunk_to_write = b''.join(self.buffer) + chunk_to_write
                        self.buffer = []
                        self.n_buffer_count = 0
                    chunk = chunk[newline + 1:]
                    total_to_write.append(chunk_to_write)
                if chunk:
                    self.buffer.append(chunk)
                    self.n_buffer_count += len(chunk)
                return total_to_write
            else:
                total_to_write = []
                while True:
                    overage = self.n_buffer_count + len(chunk) - self.type
                    if overage >= 0:
                        ret = ''.encode(self.encoding).join(self.buffer) + chunk
                        chunk_to_write = ret[:self.type]
                        chunk = ret[self.type:]
                        total_to_write.append(chunk_to_write)
                        self.buffer = []
                        self.n_buffer_count = 0
                    else:
                        self.buffer.append(chunk)
                        self.n_buffer_count += len(chunk)
                        break
                return total_to_write
        finally:
            self._buffering_lock.release()
            self.log.debug('released buffering lock for processing chunk (buffering: %d)', self.type)

    def flush(self):
        self.log.debug('acquiring buffering lock for flushing buffer')
        self._buffering_lock.acquire()
        self.log.debug('got buffering lock for flushing buffer')
        try:
            ret = ''.encode(self.encoding).join(self.buffer)
            self.buffer = []
            return ret
        finally:
            self._buffering_lock.release()
            self.log.debug('released buffering lock for flushing buffer')