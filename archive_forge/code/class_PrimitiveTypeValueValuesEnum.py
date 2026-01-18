from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PrimitiveTypeValueValuesEnum(_messages.Enum):
    """Primitive types, such as string, boolean, etc.

    Values:
      PRIMITIVE_TYPE_UNSPECIFIED: The default invalid value for a type.
      DOUBLE: A double precision number.
      STRING: An UTF-8 string.
      BOOL: A boolean value.
      TIMESTAMP: A timestamp.
      RICHTEXT: A Richtext description.
    """
    PRIMITIVE_TYPE_UNSPECIFIED = 0
    DOUBLE = 1
    STRING = 2
    BOOL = 3
    TIMESTAMP = 4
    RICHTEXT = 5