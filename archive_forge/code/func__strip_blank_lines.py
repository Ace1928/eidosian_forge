from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import enum
import re
import textwrap
def _strip_blank_lines(lines):
    """Removes lines containing only blank characters before and after the text.

  Args:
    lines: A list of lines.
  Returns:
    A list of lines without trailing or leading blank lines.
  """
    start = 0
    num_lines = len(lines)
    while lines and start < num_lines and _is_blank(lines[start]):
        start += 1
    lines = lines[start:]
    while lines and _is_blank(lines[-1]):
        lines.pop()
    return lines