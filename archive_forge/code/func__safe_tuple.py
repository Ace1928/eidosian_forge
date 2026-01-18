import collections as _collections
import dataclasses as _dataclasses
import re
import sys as _sys
import types as _types
from io import StringIO as _StringIO
def _safe_tuple(t):
    """Helper function for comparing 2-tuples"""
    return (_safe_key(t[0]), _safe_key(t[1]))