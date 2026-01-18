import collections as _collections
import dataclasses as _dataclasses
import re
import sys as _sys
import types as _types
from io import StringIO as _StringIO
def _pprint_dataclass(self, object, stream, indent, allowance, context, level):
    cls_name = object.__class__.__name__
    indent += len(cls_name) + 1
    items = [(f.name, getattr(object, f.name)) for f in _dataclasses.fields(object) if f.repr]
    stream.write(cls_name + '(')
    self._format_namespace_items(items, stream, indent, allowance, context, level)
    stream.write(')')