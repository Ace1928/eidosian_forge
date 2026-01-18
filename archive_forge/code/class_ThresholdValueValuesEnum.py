from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ThresholdValueValuesEnum(_messages.Enum):
    """Required. The harm block threshold.

    Values:
      HARM_BLOCK_THRESHOLD_UNSPECIFIED: Unspecified harm block threshold.
      BLOCK_LOW_AND_ABOVE: Block low threshold and above (i.e. block more).
      BLOCK_MEDIUM_AND_ABOVE: Block medium threshold and above.
      BLOCK_ONLY_HIGH: Block only high threshold (i.e. block less).
      BLOCK_NONE: Block none.
    """
    HARM_BLOCK_THRESHOLD_UNSPECIFIED = 0
    BLOCK_LOW_AND_ABOVE = 1
    BLOCK_MEDIUM_AND_ABOVE = 2
    BLOCK_ONLY_HIGH = 3
    BLOCK_NONE = 4