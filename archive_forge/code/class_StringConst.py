from __future__ import absolute_import
import cython
import hashlib
import operator
import os
import re
import shutil
import textwrap
from string import Template
from functools import partial
from contextlib import closing, contextmanager
from collections import defaultdict
from . import Naming
from . import Options
from . import DebugFlags
from . import StringEncoding
from .. import Utils
from .Scanning import SourceDescriptor
from ..StringIOTree import StringIOTree
class StringConst(object):
    """Global info about a C string constant held by GlobalState.
    """

    def __init__(self, cname, text, byte_string):
        self.cname = cname
        self.text = text
        self.escaped_value = StringEncoding.escape_byte_string(byte_string)
        self.py_strings = None
        self.py_versions = []

    def add_py_version(self, version):
        if not version:
            self.py_versions = [2, 3]
        elif version not in self.py_versions:
            self.py_versions.append(version)

    def get_py_string_const(self, encoding, identifier=None, is_str=False, py3str_cstring=None):
        py_strings = self.py_strings
        text = self.text
        is_str = bool(identifier or is_str)
        is_unicode = encoding is None and (not is_str)
        if encoding is None:
            encoding_key = None
        else:
            encoding = encoding.lower()
            if encoding in ('utf8', 'utf-8', 'ascii', 'usascii', 'us-ascii'):
                encoding = None
                encoding_key = None
            else:
                encoding_key = ''.join(find_alphanums(encoding))
        key = (is_str, is_unicode, encoding_key, py3str_cstring)
        if py_strings is not None:
            try:
                return py_strings[key]
            except KeyError:
                pass
        else:
            self.py_strings = {}
        if identifier:
            intern = True
        elif identifier is None:
            if isinstance(text, bytes):
                intern = bool(possible_bytes_identifier(text))
            else:
                intern = bool(possible_unicode_identifier(text))
        else:
            intern = False
        if intern:
            prefix = Naming.interned_prefixes['str']
        else:
            prefix = Naming.py_const_prefix
        if encoding_key:
            encoding_prefix = '_%s' % encoding_key
        else:
            encoding_prefix = ''
        pystring_cname = '%s%s%s_%s' % (prefix, is_str and 's' or (is_unicode and 'u') or 'b', encoding_prefix, self.cname[len(Naming.const_prefix):])
        py_string = PyStringConst(pystring_cname, encoding, is_unicode, is_str, py3str_cstring, intern)
        self.py_strings[key] = py_string
        return py_string