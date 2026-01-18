from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AllowPscValueValuesEnum(_messages.Enum):
    """Specifies whether PSC creation is allowed.

    Values:
      PSC_ALLOWED: <no description>
      PSC_BLOCKED: <no description>
      PSC_UNSPECIFIED: <no description>
    """
    PSC_ALLOWED = 0
    PSC_BLOCKED = 1
    PSC_UNSPECIFIED = 2