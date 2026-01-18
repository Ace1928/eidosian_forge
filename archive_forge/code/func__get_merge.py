from __future__ import print_function
import collections
import re
import sys
import codecs
from . import (
from .helpers import (
def _get_merge(self):
    """Parse a merge section."""
    line = self.next_line()
    if line is None:
        return None
    elif line.startswith(b'merge '):
        return line[len(b'merge '):]
    else:
        self.push_line(line)
        return None