from __future__ import absolute_import
import re
import sys
import os
import tokenize
from io import StringIO
from ._looper import looper
from .compat3 import bytes, unicode_, basestring_, next, is_unicode, coerce_text
def _interpret_if(self, parts, ns, out, defs):
    __traceback_hide__ = True
    for part in parts:
        assert not isinstance(part, basestring_)
        name, pos = (part[0], part[1])
        if name == 'else':
            result = True
        else:
            result = self._eval(part[2], ns, pos)
        if result:
            self._interpret_codes(part[3], ns, out, defs)
            break