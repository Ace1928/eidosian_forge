from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IpAllocationModeValueValuesEnum(_messages.Enum):
    """IpAllocationModeValueValuesEnum enum type.

    Values:
      ALLOCATE_IP: Allocates an internal IPv4 IP address from subnets
        secondary IP Range.
      DO_NOT_ALLOCATE_IP: No IP allocation is done for the subinterface.
      UNSPECIFIED: <no description>
    """
    ALLOCATE_IP = 0
    DO_NOT_ALLOCATE_IP = 1
    UNSPECIFIED = 2