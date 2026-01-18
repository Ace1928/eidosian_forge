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
class FilterUserInjector(logging.Filter):
    """
    This is a filter which injects the current user as the 'user' attribute on each record. We need to add this filter
    to all logger handlers so that 3rd party libraries won't print an exception due to user not being defined.
    """
    try:
        username = getpass.getuser()
    except KeyError:
        username = 'uid=%s' % os.getuid()

    def filter(self, record):
        record.user = FilterUserInjector.username
        return True