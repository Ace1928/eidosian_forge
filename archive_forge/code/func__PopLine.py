import encodings.raw_unicode_escape  # pylint: disable=unused-import
import encodings.unicode_escape  # pylint: disable=unused-import
import io
import math
import re
from google.protobuf.internal import decoder
from google.protobuf.internal import type_checkers
from google.protobuf import descriptor
from google.protobuf import text_encoding
from google.protobuf import unknown_fields
def _PopLine(self):
    while len(self._current_line) <= self._column:
        try:
            self._current_line = next(self._lines)
        except StopIteration:
            self._current_line = ''
            self._more_lines = False
            return
        else:
            self._line += 1
            self._column = 0