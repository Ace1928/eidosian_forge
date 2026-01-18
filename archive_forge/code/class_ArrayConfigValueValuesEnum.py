from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ArrayConfigValueValuesEnum(_messages.Enum):
    """Indicates that this field supports operations on `array_value`s.

    Values:
      ARRAY_CONFIG_UNSPECIFIED: The index does not support additional array
        queries.
      CONTAINS: The index supports array containment queries.
    """
    ARRAY_CONFIG_UNSPECIFIED = 0
    CONTAINS = 1