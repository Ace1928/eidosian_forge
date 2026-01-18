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
def _aggregate_keywords(keywords, sep, prefix, raw=False):
    """take our keyword arguments, and a separator, and compose the list of
    flat long (and short) arguments.  example

        {'color': 'never', 't': True, 'something': True} with sep '='

    becomes

        ['--color=never', '-t', '--something']

    the `raw` argument indicates whether or not we should leave the argument
    name alone, or whether we should replace "_" with "-".  if we pass in a
    dictionary, like this:

        sh.command({"some_option": 12})

    then `raw` gets set to True, because we want to leave the key as-is, to
    produce:

        ['--some_option=12']

    but if we just use a command's kwargs, `raw` is False, which means this:

        sh.command(some_option=12)

    becomes:

        ['--some-option=12']

    essentially, using kwargs is a convenience, but it lacks the ability to
    put a '-' in the name, so we do the replacement of '_' to '-' for you.
    but when you really don't want that to happen, you should use a
    dictionary instead with the exact names you want
    """
    processed = []
    for k, maybe_list_of_v in keywords.items():
        list_of_v = [maybe_list_of_v]
        if isinstance(maybe_list_of_v, (list, tuple)):
            list_of_v = maybe_list_of_v
        for v in list_of_v:
            if len(k) == 1:
                if v is not False:
                    processed.append('-' + k)
                    if v is not True:
                        processed.append(str(v))
            else:
                if not raw:
                    k = k.replace('_', '-')
                if v is True:
                    processed.append(prefix + k)
                elif v is False:
                    pass
                elif sep is None or sep == ' ':
                    processed.append(prefix + k)
                    processed.append(str(v))
                else:
                    arg = f'{prefix}{k}{sep}{v}'
                    processed.append(arg)
    return processed