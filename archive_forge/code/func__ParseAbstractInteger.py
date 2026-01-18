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
def _ParseAbstractInteger(text):
    """Parses an integer without checking size/signedness.

  Args:
    text: The text to parse.

  Returns:
    The integer value.

  Raises:
    ValueError: Thrown Iff the text is not a valid integer.
  """
    orig_text = text
    c_octal_match = re.match('(-?)0(\\d+)$', text)
    if c_octal_match:
        text = c_octal_match.group(1) + '0o' + c_octal_match.group(2)
    try:
        return int(text, 0)
    except ValueError:
        raise ValueError("Couldn't parse integer: %s" % orig_text)