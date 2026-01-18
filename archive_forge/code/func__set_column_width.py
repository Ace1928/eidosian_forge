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
def _set_column_width(self) -> None:
    if os.isatty(1):
        tty_size = unpack('HHHH', fcntl.ioctl(1, termios.TIOCGWINSZ, pack('HHHH', 0, 0, 0, 0)))[1]
    else:
        tty_size = 0
    self.columns = max(79, tty_size - 1)