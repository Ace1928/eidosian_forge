from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AllowPrivateGoogleAccessValueValuesEnum(_messages.Enum):
    """Specifies whether private Google access is allowed.

    Values:
      PRIVATE_GOOGLE_ACCESS_ALLOWED: <no description>
      PRIVATE_GOOGLE_ACCESS_BLOCKED: <no description>
      PRIVATE_GOOGLE_ACCESS_UNSPECIFIED: <no description>
    """
    PRIVATE_GOOGLE_ACCESS_ALLOWED = 0
    PRIVATE_GOOGLE_ACCESS_BLOCKED = 1
    PRIVATE_GOOGLE_ACCESS_UNSPECIFIED = 2