from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HighSensitivityMappingValueValuesEnum(_messages.Enum):
    """Resource value mapping for high-sensitivity Sensitive Data Protection
    findings

    Values:
      RESOURCE_VALUE_UNSPECIFIED: Unspecific value
      HIGH: High resource value
      MEDIUM: Medium resource value
      LOW: Low resource value
      NONE: No resource value, e.g. ignore these resources
    """
    RESOURCE_VALUE_UNSPECIFIED = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3
    NONE = 4