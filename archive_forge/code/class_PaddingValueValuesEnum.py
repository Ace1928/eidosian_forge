from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PaddingValueValuesEnum(_messages.Enum):
    """The amount of padding around the widget

    Values:
      PADDING_SIZE_UNSPECIFIED: No padding size specified, will default to
        P_EXTRA_SMALL
      P_EXTRA_SMALL: Extra small padding
      P_SMALL: Small padding
      P_MEDIUM: Medium padding
      P_LARGE: Large padding
      P_EXTRA_LARGE: Extra large padding
    """
    PADDING_SIZE_UNSPECIFIED = 0
    P_EXTRA_SMALL = 1
    P_SMALL = 2
    P_MEDIUM = 3
    P_LARGE = 4
    P_EXTRA_LARGE = 5