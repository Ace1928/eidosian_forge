from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AllowInterconnectValueValuesEnum(_messages.Enum):
    """Specifies whether Cloud Interconnect creation is allowed.

    Values:
      INTERCONNECT_ALLOWED: <no description>
      INTERCONNECT_BLOCKED: <no description>
      INTERCONNECT_UNSPECIFIED: <no description>
    """
    INTERCONNECT_ALLOWED = 0
    INTERCONNECT_BLOCKED = 1
    INTERCONNECT_UNSPECIFIED = 2