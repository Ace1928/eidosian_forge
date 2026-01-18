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
def _ConsumeUint64(tokenizer):
    """Consumes an unsigned 64bit integer number from tokenizer.

  Args:
    tokenizer: A tokenizer used to parse the number.

  Returns:
    The integer parsed.

  Raises:
    ParseError: If an unsigned 64bit integer couldn't be consumed.
  """
    return _ConsumeInteger(tokenizer, is_signed=False, is_long=True)