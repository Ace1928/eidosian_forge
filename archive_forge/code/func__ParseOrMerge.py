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
def _ParseOrMerge(self, lines, message):
    """Converts a text representation of a protocol message into a message.

    Args:
      lines: Lines of a message's text representation.
      message: A protocol buffer message to merge into.

    Raises:
      ParseError: On text parsing problems.
    """
    try:
        str_lines = (line if isinstance(line, str) else line.decode('utf-8') for line in lines)
        tokenizer = Tokenizer(str_lines)
    except UnicodeDecodeError as e:
        raise ParseError from e
    if message:
        self.root_type = message.DESCRIPTOR.full_name
    while not tokenizer.AtEnd():
        self._MergeField(tokenizer, message)