import codecs
import io
import os
import re
import sys
import typing as t
from weakref import WeakKeyDictionary
def _safe_write(s):
    try:
        return _write(s)
    except BaseException:
        ansi_wrapper.reset_all()
        raise