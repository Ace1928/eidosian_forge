from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DirectionValueValuesEnum(_messages.Enum):
    """Output only. Direction of the routes exchanged with the peer network,
    from the VMware Engine network perspective: * Routes of direction
    `INCOMING` are imported from the peer network. * Routes of direction
    `OUTGOING` are exported from the intranet VPC network of the VMware Engine
    network.

    Values:
      DIRECTION_UNSPECIFIED: Unspecified exchanged routes direction. This is
        default.
      INCOMING: Routes imported from the peer network.
      OUTGOING: Routes exported to the peer network.
    """
    DIRECTION_UNSPECIFIED = 0
    INCOMING = 1
    OUTGOING = 2