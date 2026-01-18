import collections as _collections
import dataclasses as _dataclasses
import re
import sys as _sys
import types as _types
from io import StringIO as _StringIO
def _format_dict_items(self, items, stream, indent, allowance, context, level):
    write = stream.write
    indent += self._indent_per_level
    delimnl = ',\n' + ' ' * indent
    last_index = len(items) - 1
    for i, (key, ent) in enumerate(items):
        last = i == last_index
        rep = self._repr(key, context, level)
        write(rep)
        write(': ')
        self._format(ent, stream, indent + len(rep) + 2, allowance if last else 1, context, level)
        if not last:
            write(delimnl)