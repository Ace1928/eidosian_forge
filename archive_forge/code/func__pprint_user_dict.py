import collections as _collections
import dataclasses as _dataclasses
import re
import sys as _sys
import types as _types
from io import StringIO as _StringIO
def _pprint_user_dict(self, object, stream, indent, allowance, context, level):
    self._format(object.data, stream, indent, allowance, context, level - 1)