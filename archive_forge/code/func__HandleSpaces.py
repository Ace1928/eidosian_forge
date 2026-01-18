from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import fnmatch
import os
import re
def _HandleSpaces(line):
    """Handles spaces in a line.

  In particular, deals with trailing spaces (which are stripped unless
  escaped) and escaped spaces throughout the line (which are unescaped).

  Args:
    line: str, the line

  Returns:
    str, the line with spaces handled
  """

    def _Rstrip(line):
        """Strips unescaped trailing spaces."""
        tokens = []
        i = 0
        while i < len(line):
            curr = line[i]
            if curr == '\\':
                if i + 1 >= len(line):
                    tokens.append(curr)
                    break
                tokens.append(curr + line[i + 1])
                i += 2
            else:
                tokens.append(curr)
                i += 1
        res = []
        only_seen_spaces = True
        for curr in reversed(tokens):
            if only_seen_spaces and curr == ' ':
                continue
            only_seen_spaces = False
            res.append(curr)
        return ''.join(reversed(res))

    def _UnescapeSpaces(line):
        """Unescapes all spaces in a line."""
        return line.replace('\\ ', ' ')
    return _UnescapeSpaces(_Rstrip(line))