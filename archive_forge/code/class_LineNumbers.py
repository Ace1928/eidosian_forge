import bisect
import re
from typing import Dict, List, Tuple
class LineNumbers(object):
    """
  Class to convert between character offsets in a text string, and pairs (line, column) of 1-based
  line and 0-based column numbers, as used by tokens and AST nodes.

  This class expects unicode for input and stores positions in unicode. But it supports
  translating to and from utf8 offsets, which are used by ast parsing.
  """

    def __init__(self, text):
        self._line_offsets = [m.start(0) for m in _line_start_re.finditer(text)]
        self._text = text
        self._text_len = len(text)
        self._utf8_offset_cache = {}

    def from_utf8_col(self, line, utf8_column):
        """
    Given a 1-based line number and 0-based utf8 column, returns a 0-based unicode column.
    """
        offsets = self._utf8_offset_cache.get(line)
        if offsets is None:
            end_offset = self._line_offsets[line] if line < len(self._line_offsets) else self._text_len
            line_text = self._text[self._line_offsets[line - 1]:end_offset]
            offsets = [i for i, c in enumerate(line_text) for byte in c.encode('utf8')]
            offsets.append(len(line_text))
            self._utf8_offset_cache[line] = offsets
        return offsets[max(0, min(len(offsets) - 1, utf8_column))]

    def line_to_offset(self, line, column):
        """
    Converts 1-based line number and 0-based column to 0-based character offset into text.
    """
        line -= 1
        if line >= len(self._line_offsets):
            return self._text_len
        elif line < 0:
            return 0
        else:
            return min(self._line_offsets[line] + max(0, column), self._text_len)

    def offset_to_line(self, offset):
        """
    Converts 0-based character offset to pair (line, col) of 1-based line and 0-based column
    numbers.
    """
        offset = max(0, min(self._text_len, offset))
        line_index = bisect.bisect_right(self._line_offsets, offset) - 1
        return (line_index + 1, offset - self._line_offsets[line_index])