from __future__ import print_function
import collections
import re
import sys
import codecs
from . import (
from .helpers import (
def _name_value(self, s):
    """Parse a (name,value) tuple from 'name value-length value'."""
    parts = s.split(b' ', 2)
    name = parts[0]
    if len(parts) == 1:
        value = None
    else:
        size = int(parts[1])
        value = parts[2]
        still_to_read = size - len(value)
        if still_to_read > 0:
            read_bytes = self.read_bytes(still_to_read)
            value += b'\n' + read_bytes[:still_to_read - 1]
    return (name, value)