import collections as _collections
import dataclasses as _dataclasses
import re
import sys as _sys
import types as _types
from io import StringIO as _StringIO
def _pprint_mappingproxy(self, object, stream, indent, allowance, context, level):
    stream.write('mappingproxy(')
    self._format(object.copy(), stream, indent + 13, allowance + 1, context, level)
    stream.write(')')