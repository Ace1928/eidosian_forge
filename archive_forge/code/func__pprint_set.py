import collections as _collections
import dataclasses as _dataclasses
import re
import sys as _sys
import types as _types
from io import StringIO as _StringIO
def _pprint_set(self, object, stream, indent, allowance, context, level):
    if not len(object):
        stream.write(repr(object))
        return
    typ = object.__class__
    if typ is set:
        stream.write('{')
        endchar = '}'
    else:
        stream.write(typ.__name__ + '({')
        endchar = '})'
        indent += len(typ.__name__) + 1
    object = sorted(object, key=_safe_key)
    self._format_items(object, stream, indent, allowance + len(endchar), context, level)
    stream.write(endchar)