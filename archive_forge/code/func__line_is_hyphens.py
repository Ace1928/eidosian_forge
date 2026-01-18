from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import enum
import re
import textwrap
def _line_is_hyphens(line):
    """Returns whether the line is entirely hyphens (and not blank)."""
    return line and (not line.strip('-'))