from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConfidenceValueValuesEnum(_messages.Enum):
    """Filtered category

    Values:
      CONFIDENCE_UNSPECIFIED: <no description>
      CONFIDENCE_LOW: <no description>
      CONFIDENCE_MEDIUM: <no description>
      CONFIDENCE_HIGH: <no description>
    """
    CONFIDENCE_UNSPECIFIED = 0
    CONFIDENCE_LOW = 1
    CONFIDENCE_MEDIUM = 2
    CONFIDENCE_HIGH = 3