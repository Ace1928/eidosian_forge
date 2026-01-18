from __future__ import print_function
import collections
import re
import sys
import codecs
from . import (
from .helpers import (
def _get_from(self, required_for=None):
    """Parse a from section."""
    line = self.next_line()
    if line is None:
        return None
    elif line.startswith(b'from '):
        return line[len(b'from '):]
    elif required_for:
        self.abort(errors.MissingSection, required_for, 'from')
    else:
        self.push_line(line)
        return None