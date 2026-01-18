from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IpTypeValueValuesEnum(_messages.Enum):
    """Output only. The type of this network attachment.

    Values:
      IP_TYPE_UNSPECIFIED: The type of this ip is unknown.
      FIXED: The ip address is fixed.
      DYNAMIC: The ip address is dynamic.
    """
    IP_TYPE_UNSPECIFIED = 0
    FIXED = 1
    DYNAMIC = 2