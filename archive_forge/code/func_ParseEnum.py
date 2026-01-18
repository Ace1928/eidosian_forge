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
def ParseEnum(field, value):
    """Parse an enum value.

  The value can be specified by a number (the enum value), or by
  a string literal (the enum name).

  Args:
    field: Enum field descriptor.
    value: String value.

  Returns:
    Enum value number.

  Raises:
    ValueError: If the enum value could not be parsed.
  """
    enum_descriptor = field.enum_type
    try:
        number = int(value, 0)
    except ValueError:
        enum_value = enum_descriptor.values_by_name.get(value, None)
        if enum_value is None:
            raise ValueError('Enum type "%s" has no value named %s.' % (enum_descriptor.full_name, value))
    else:
        if not field.enum_type.is_closed:
            return number
        enum_value = enum_descriptor.values_by_number.get(number, None)
        if enum_value is None:
            raise ValueError('Enum type "%s" has no value with number %d.' % (enum_descriptor.full_name, number))
    return enum_value.number