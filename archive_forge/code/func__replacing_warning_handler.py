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
def _replacing_warning_handler(self, exception: UnicodeError) -> tuple[str | bytes, int]:
    self.deprecated('Non UTF-8 encoded data replaced with "?" while displaying text to stdout/stderr, this is temporary and will become an error', version='2.18')
    return ('?', exception.end)