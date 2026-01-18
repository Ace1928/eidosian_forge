from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class IamValueValuesEnum(_messages.Enum):
    """Trusted attributes supplied by the IAM system.

    Values:
      NO_ATTR: Default non-attribute.
      AUTHORITY: Either principal or (if present) authority
      ATTRIBUTION: selector Always the original principal, but making clear
    """
    NO_ATTR = 0
    AUTHORITY = 1
    ATTRIBUTION = 2