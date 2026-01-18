import codecs
import io
import os
import re
import sys
import typing as t
from weakref import WeakKeyDictionary
def _get_argv_encoding() -> str:
    return getattr(sys.stdin, 'encoding', None) or sys.getfilesystemencoding()