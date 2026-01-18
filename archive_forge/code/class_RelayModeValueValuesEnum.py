from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RelayModeValueValuesEnum(_messages.Enum):
    """Method used to make Relay available

    Values:
      RELAY_MODE_UNSPECIFIED: Default value. This shouldn't be used.
      DISABLED: disabled
      INTERNAL_VPC_LB: exposed via internal load balancer
      EXTERNAL_LB: exposed via external load balancer
    """
    RELAY_MODE_UNSPECIFIED = 0
    DISABLED = 1
    INTERNAL_VPC_LB = 2
    EXTERNAL_LB = 3