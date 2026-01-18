from __future__ import absolute_import
import re
import sys
import os
import tokenize
from io import StringIO
from ._looper import looper
from .compat3 import bytes, unicode_, basestring_, next, is_unicode, coerce_text
def _interpret_for(self, vars, expr, content, ns, out, defs):
    __traceback_hide__ = True
    for item in expr:
        if len(vars) == 1:
            ns[vars[0]] = item
        else:
            if len(vars) != len(item):
                raise ValueError('Need %i items to unpack (got %i items)' % (len(vars), len(item)))
            for name, value in zip(vars, item):
                ns[name] = value
        try:
            self._interpret_codes(content, ns, out, defs)
        except _TemplateContinue:
            continue
        except _TemplateBreak:
            break