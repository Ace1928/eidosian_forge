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
@classmethod
def _extract_call_args(cls, kwargs):
    """takes kwargs that were passed to a command's __call__ and extracts
        out the special keyword arguments, we return a tuple of special keyword
        args, and kwargs that will go to the exec'ed command"""
    kwargs = kwargs.copy()
    call_args = {}
    for parg, default in cls._call_args.items():
        key = '_' + parg
        if key in kwargs:
            call_args[parg] = kwargs[key]
            del kwargs[key]
    merged_args = cls._call_args.copy()
    merged_args.update(call_args)
    invalid_kwargs = special_kwarg_validator(call_args, merged_args, cls._kwarg_validators)
    if invalid_kwargs:
        exc_msg = []
        for kwarg, error_msg in invalid_kwargs:
            exc_msg.append(f'  {kwarg!r}: {error_msg}')
        exc_msg = '\n'.join(exc_msg)
        raise TypeError(f'Invalid special arguments:\n\n{exc_msg}\n')
    return (call_args, kwargs)