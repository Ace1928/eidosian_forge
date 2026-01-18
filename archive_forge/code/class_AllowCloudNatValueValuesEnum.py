from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AllowCloudNatValueValuesEnum(_messages.Enum):
    """Specifies whether cloud NAT creation is allowed.

    Values:
      CLOUD_NAT_ALLOWED: <no description>
      CLOUD_NAT_BLOCKED: <no description>
      CLOUD_NAT_UNSPECIFIED: <no description>
    """
    CLOUD_NAT_ALLOWED = 0
    CLOUD_NAT_BLOCKED = 1
    CLOUD_NAT_UNSPECIFIED = 2