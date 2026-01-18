from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UnsatisfiedReasonValueValuesEnum(_messages.Enum):
    """Indicates the reason why the VPN connection does not meet the high
    availability redundancy criteria/requirement. Valid values is
    INCOMPLETE_TUNNELS_COVERAGE.

    Values:
      INCOMPLETE_TUNNELS_COVERAGE: <no description>
    """
    INCOMPLETE_TUNNELS_COVERAGE = 0