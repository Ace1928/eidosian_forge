from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DecimalTargetTypesValueListEntryValuesEnum(_messages.Enum):
    """DecimalTargetTypesValueListEntryValuesEnum enum type.

    Values:
      DECIMAL_TARGET_TYPE_UNSPECIFIED: Invalid type.
      NUMERIC: Decimal values could be converted to NUMERIC type.
      BIGNUMERIC: Decimal values could be converted to BIGNUMERIC type.
      STRING: Decimal values could be converted to STRING type.
    """
    DECIMAL_TARGET_TYPE_UNSPECIFIED = 0
    NUMERIC = 1
    BIGNUMERIC = 2
    STRING = 3