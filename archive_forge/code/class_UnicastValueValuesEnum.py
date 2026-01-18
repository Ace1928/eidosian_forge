from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UnicastValueValuesEnum(_messages.Enum):
    """Specifies which type of unicast is supported.

    Values:
      UNICAST_SDN: <no description>
      UNICAST_ULL: <no description>
      UNICAST_UNSPECIFIED: <no description>
    """
    UNICAST_SDN = 0
    UNICAST_ULL = 1
    UNICAST_UNSPECIFIED = 2