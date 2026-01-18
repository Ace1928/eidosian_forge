from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Ipv6EndpointTypeValueValuesEnum(_messages.Enum):
    """The endpoint type of this address, which should be VM or NETLB. This
    is used for deciding which type of endpoint this address can be used after
    the external IPv6 address reservation.

    Values:
      NETLB: Reserved IPv6 address can be used on network load balancer.
      VM: Reserved IPv6 address can be used on VM.
    """
    NETLB = 0
    VM = 1