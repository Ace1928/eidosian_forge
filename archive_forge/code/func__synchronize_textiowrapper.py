from __future__ import annotations
import collections.abc as c
import codecs
import ctypes.util
import fcntl
import getpass
import io
import logging
import os
import random
import subprocess
import sys
import termios
import textwrap
import threading
import time
import tty
import typing as t
from functools import wraps
from struct import unpack, pack
from ansible import constants as C
from ansible.errors import AnsibleError, AnsibleAssertionError, AnsiblePromptInterrupt, AnsiblePromptNoninteractive
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.module_utils.six import text_type
from ansible.utils.color import stringc
from ansible.utils.multiprocessing import context as multiprocessing_context
from ansible.utils.singleton import Singleton
from ansible.utils.unsafe_proxy import wrap_var
def _synchronize_textiowrapper(tio: t.TextIO, lock: threading.RLock):

    def _wrap_with_lock(f, lock):

        @wraps(f)
        def locking_wrapper(*args, **kwargs):
            with lock:
                return f(*args, **kwargs)
        return locking_wrapper
    buffer = tio.buffer
    buffer.write = _wrap_with_lock(buffer.write, lock)
    buffer.flush = _wrap_with_lock(buffer.flush, lock)