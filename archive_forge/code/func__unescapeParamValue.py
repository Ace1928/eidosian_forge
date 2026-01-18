import errno
import operator
import os
import random
import re
import shlex
import socket
import stat
import string
import struct
import sys
import textwrap
import time
import traceback
from functools import reduce
from os import path
from typing import Optional
from twisted.internet import protocol, reactor, task
from twisted.persisted import styles
from twisted.protocols import basic
from twisted.python import _textattributes, log, reflect
@classmethod
def _unescapeParamValue(cls, value):
    """
        Unescape an ISUPPORT parameter.

        The only form of supported escape is C{\\xHH}, where HH must be a valid
        2-digit hexadecimal number.

        @rtype: C{str}
        """

    def _unescape():
        parts = value.split('\\x')
        yield parts.pop(0)
        for s in parts:
            octet, rest = (s[:2], s[2:])
            try:
                octet = int(octet, 16)
            except ValueError:
                raise ValueError(f'Invalid hex octet: {octet!r}')
            yield (chr(octet) + rest)
    if '\\x' not in value:
        return value
    return ''.join(_unescape())