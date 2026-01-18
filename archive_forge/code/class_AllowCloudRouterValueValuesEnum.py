from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AllowCloudRouterValueValuesEnum(_messages.Enum):
    """Specifies whether cloud router creation is allowed.

    Values:
      CLOUD_ROUTER_ALLOWED: <no description>
      CLOUD_ROUTER_BLOCKED: <no description>
      CLOUD_ROUTER_UNSPECIFIED: <no description>
    """
    CLOUD_ROUTER_ALLOWED = 0
    CLOUD_ROUTER_BLOCKED = 1
    CLOUD_ROUTER_UNSPECIFIED = 2