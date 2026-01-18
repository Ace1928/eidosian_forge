from __future__ import absolute_import
import re
import sys
import os
import tokenize
from io import StringIO
from ._looper import looper
from .compat3 import bytes, unicode_, basestring_, next, is_unicode, coerce_text
def _interpret(self, ns):
    __traceback_hide__ = True
    parts = []
    defs = {}
    self._interpret_codes(self._parsed, ns, out=parts, defs=defs)
    if '__inherit__' in defs:
        inherit = defs.pop('__inherit__')
    else:
        inherit = None
    return (''.join(parts), defs, inherit)