from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AllowedSubnetStackTypesValueListEntryValuesEnum(_messages.Enum):
    """AllowedSubnetStackTypesValueListEntryValuesEnum enum type.

    Values:
      SUBNET_STACK_TYPE_IPV4_IPV6: <no description>
      SUBNET_STACK_TYPE_IPV4_ONLY: <no description>
      SUBNET_STACK_TYPE_IPV6_ONLY: <no description>
      SUBNET_STACK_TYPE_UNSPECIFIED: <no description>
    """
    SUBNET_STACK_TYPE_IPV4_IPV6 = 0
    SUBNET_STACK_TYPE_IPV4_ONLY = 1
    SUBNET_STACK_TYPE_IPV6_ONLY = 2
    SUBNET_STACK_TYPE_UNSPECIFIED = 3