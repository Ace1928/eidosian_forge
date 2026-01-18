import re
import sys
import warnings
from bs4.css import CSS
from bs4.formatter import (
def _normalize_search_value(self, value):
    if isinstance(value, str) or isinstance(value, Callable) or hasattr(value, 'match') or isinstance(value, bool) or (value is None):
        return value
    if isinstance(value, bytes):
        return value.decode('utf8')
    if hasattr(value, '__iter__'):
        new_value = []
        for v in value:
            if hasattr(v, '__iter__') and (not isinstance(v, bytes)) and (not isinstance(v, str)):
                new_value.append(v)
            else:
                new_value.append(self._normalize_search_value(v))
        return new_value
    return str(str(value))