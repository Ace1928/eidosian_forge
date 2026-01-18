from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NatPolicyValueValuesEnum(_messages.Enum):
    """Must have a value of NO_NAT. Protocol forwarding delivers packets
    while preserving the destination IP address of the forwarding rule
    referencing the target instance.

    Values:
      NO_NAT: No NAT performed.
    """
    NO_NAT = 0