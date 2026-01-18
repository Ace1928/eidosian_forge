from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AllowMultiNicInSameNetworkValueValuesEnum(_messages.Enum):
    """Specifies whether multi-nic in the same network is allowed.

    Values:
      MULTI_NIC_IN_SAME_NETWORK_ALLOWED: <no description>
      MULTI_NIC_IN_SAME_NETWORK_BLOCKED: <no description>
      MULTI_NIC_IN_SAME_NETWORK_UNSPECIFIED: <no description>
    """
    MULTI_NIC_IN_SAME_NETWORK_ALLOWED = 0
    MULTI_NIC_IN_SAME_NETWORK_BLOCKED = 1
    MULTI_NIC_IN_SAME_NETWORK_UNSPECIFIED = 2