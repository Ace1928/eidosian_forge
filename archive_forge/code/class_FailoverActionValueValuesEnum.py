from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FailoverActionValueValuesEnum(_messages.Enum):
    """The action to perform in case of zone failure. Only one value is
    supported, NO_FAILOVER. The default is NO_FAILOVER.

    Values:
      NO_FAILOVER: <no description>
      UNKNOWN: <no description>
    """
    NO_FAILOVER = 0
    UNKNOWN = 1