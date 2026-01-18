from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import fnmatch
import os
import re
def _Unescape(line):
    """Unescapes a line.

  The escape character is '\\'. An escaped backslash turns into one backslash;
  any other escaped character is ignored.

  Args:
    line: str, the line to unescape

  Returns:
    str, the unescaped line

  """
    return re.sub('\\\\([^\\\\])', '\\1', line).replace('\\\\', '\\')