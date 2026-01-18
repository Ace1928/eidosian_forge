import collections as _collections
import dataclasses as _dataclasses
import re
import sys as _sys
import types as _types
from io import StringIO as _StringIO
def _pprint_dict(self, object, stream, indent, allowance, context, level):
    write = stream.write
    write('{')
    if self._indent_per_level > 1:
        write((self._indent_per_level - 1) * ' ')
    length = len(object)
    if length:
        if self._sort_dicts:
            items = sorted(object.items(), key=_safe_tuple)
        else:
            items = object.items()
        self._format_dict_items(items, stream, indent, allowance + 1, context, level)
    write('}')