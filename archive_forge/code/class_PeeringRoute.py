from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PeeringRoute(_messages.Message):
    """Exchanged network peering route.

  Enums:
    DirectionValueValuesEnum: Output only. Direction of the routes exchanged
      with the peer network, from the VMware Engine network perspective: *
      Routes of direction `INCOMING` are imported from the peer network. *
      Routes of direction `OUTGOING` are exported from the intranet VPC
      network of the VMware Engine network.
    TypeValueValuesEnum: Output only. Type of the route in the peer VPC
      network.

  Fields:
    destRange: Output only. Destination range of the peering route in CIDR
      notation.
    direction: Output only. Direction of the routes exchanged with the peer
      network, from the VMware Engine network perspective: * Routes of
      direction `INCOMING` are imported from the peer network. * Routes of
      direction `OUTGOING` are exported from the intranet VPC network of the
      VMware Engine network.
    imported: Output only. True if the peering route has been imported from a
      peered VPC network; false otherwise. The import happens if the field
      `NetworkPeering.importCustomRoutes` is true for this network,
      `NetworkPeering.exportCustomRoutes` is true for the peer VPC network,
      and the import does not result in a route conflict.
    nextHopRegion: Output only. Region containing the next hop of the peering
      route. This field only applies to dynamic routes in the peer VPC
      network.
    priority: Output only. The priority of the peering route.
    type: Output only. Type of the route in the peer VPC network.
  """

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

    class TypeValueValuesEnum(_messages.Enum):
        """Output only. Type of the route in the peer VPC network.

    Values:
      TYPE_UNSPECIFIED: Unspecified peering route type. This is the default
        value.
      DYNAMIC_PEERING_ROUTE: Dynamic routes in the peer network.
      STATIC_PEERING_ROUTE: Static routes in the peer network.
      SUBNET_PEERING_ROUTE: Created, updated, and removed automatically by
        Google Cloud when subnets are created, modified, or deleted in the
        peer network.
    """
        TYPE_UNSPECIFIED = 0
        DYNAMIC_PEERING_ROUTE = 1
        STATIC_PEERING_ROUTE = 2
        SUBNET_PEERING_ROUTE = 3
    destRange = _messages.StringField(1)
    direction = _messages.EnumField('DirectionValueValuesEnum', 2)
    imported = _messages.BooleanField(3)
    nextHopRegion = _messages.StringField(4)
    priority = _messages.IntegerField(5)
    type = _messages.EnumField('TypeValueValuesEnum', 6)