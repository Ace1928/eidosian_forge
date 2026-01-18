from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AllowVpnValueValuesEnum(_messages.Enum):
    """Specifies whether VPN creation is allowed.

    Values:
      VPN_ALLOWED: <no description>
      VPN_BLOCKED: <no description>
      VPN_UNSPECIFIED: <no description>
    """
    VPN_ALLOWED = 0
    VPN_BLOCKED = 1
    VPN_UNSPECIFIED = 2