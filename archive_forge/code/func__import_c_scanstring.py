from __future__ import absolute_import
import re
import sys
import struct
from .compat import PY3, unichr
from .scanner import make_scanner, JSONDecodeError
def _import_c_scanstring():
    try:
        from ._speedups import scanstring
        return scanstring
    except ImportError:
        return None