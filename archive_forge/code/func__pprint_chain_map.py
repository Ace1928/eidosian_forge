import collections as _collections
import dataclasses as _dataclasses
import re
import sys as _sys
import types as _types
from io import StringIO as _StringIO
def _pprint_chain_map(self, object, stream, indent, allowance, context, level):
    if not len(object.maps):
        stream.write(repr(object))
        return
    cls = object.__class__
    stream.write(cls.__name__ + '(')
    indent += len(cls.__name__) + 1
    for i, m in enumerate(object.maps):
        if i == len(object.maps) - 1:
            self._format(m, stream, indent, allowance + 1, context, level)
            stream.write(')')
        else:
            self._format(m, stream, indent, 1, context, level)
            stream.write(',\n' + ' ' * indent)