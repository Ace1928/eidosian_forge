import collections as _collections
import dataclasses as _dataclasses
import re
import sys as _sys
import types as _types
from io import StringIO as _StringIO
def _pprint_simplenamespace(self, object, stream, indent, allowance, context, level):
    if type(object) is _types.SimpleNamespace:
        cls_name = 'namespace'
    else:
        cls_name = object.__class__.__name__
    indent += len(cls_name) + 1
    items = object.__dict__.items()
    stream.write(cls_name + '(')
    self._format_namespace_items(items, stream, indent, allowance, context, level)
    stream.write(')')