import sys
import json
from .symbols import *
from .symbols import Symbol
def _obj_diff(self, a, b):
    if a is b:
        return (self.options.syntax.emit_value_diff(a, b, 1.0), 1.0)
    if isinstance(a, dict) and isinstance(b, dict):
        return self._dict_diff(a, b)
    elif isinstance(a, tuple) and isinstance(b, tuple):
        return self._list_diff(a, b)
    elif isinstance(a, list) and isinstance(b, list):
        return self._list_diff(a, b)
    elif isinstance(a, set) and isinstance(b, set):
        return self._set_diff(a, b)
    elif a != b:
        return (self.options.syntax.emit_value_diff(a, b, 0.0), 0.0)
    else:
        return (self.options.syntax.emit_value_diff(a, b, 1.0), 1.0)