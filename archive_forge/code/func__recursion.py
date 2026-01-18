import collections as _collections
import dataclasses as _dataclasses
import re
import sys as _sys
import types as _types
from io import StringIO as _StringIO
def _recursion(object):
    return '<Recursion on %s with id=%s>' % (type(object).__name__, id(object))