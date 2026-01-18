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
def Merge(text, message, allow_unknown_extension=False, allow_field_number=False, descriptor_pool=None, allow_unknown_field=False):
    """Parses a text representation of a protocol message into a message.

  Like Parse(), but allows repeated values for a non-repeated field, and uses
  the last one. This means any non-repeated, top-level fields specified in text
  replace those in the message.

  Args:
    text (str): Message text representation.
    message (Message): A protocol buffer message to merge into.
    allow_unknown_extension: if True, skip over missing extensions and keep
      parsing
    allow_field_number: if True, both field number and field name are allowed.
    descriptor_pool (DescriptorPool): Descriptor pool used to resolve Any types.
    allow_unknown_field: if True, skip over unknown field and keep
      parsing. Avoid to use this option if possible. It may hide some
      errors (e.g. spelling error on field name)

  Returns:
    Message: The same message passed as argument.

  Raises:
    ParseError: On text parsing problems.
  """
    return MergeLines(text.split(b'\n' if isinstance(text, bytes) else u'\n'), message, allow_unknown_extension, allow_field_number, descriptor_pool=descriptor_pool, allow_unknown_field=allow_unknown_field)