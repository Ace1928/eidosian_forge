from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExchangedPeeringRoute(_messages.Message):
    """A ExchangedPeeringRoute object.

  Enums:
    TypeValueValuesEnum: The type of the peering route.

  Fields:
    destRange: The destination range of the route.
    imported: True if the peering route has been imported from a peer. The
      actual import happens if the field networkPeering.importCustomRoutes is
      true for this network, and networkPeering.exportCustomRoutes is true for
      the peer network, and the import does not result in a route conflict.
    nextHopRegion: The region of peering route next hop, only applies to
      dynamic routes.
    priority: The priority of the peering route.
    type: The type of the peering route.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """The type of the peering route.

    Values:
      DYNAMIC_PEERING_ROUTE: For routes exported from local network.
      STATIC_PEERING_ROUTE: The peering route.
      SUBNET_PEERING_ROUTE: The peering route corresponding to subnetwork
        range.
    """
        DYNAMIC_PEERING_ROUTE = 0
        STATIC_PEERING_ROUTE = 1
        SUBNET_PEERING_ROUTE = 2
    destRange = _messages.StringField(1)
    imported = _messages.BooleanField(2)
    nextHopRegion = _messages.StringField(3)
    priority = _messages.IntegerField(4, variant=_messages.Variant.UINT32)
    type = _messages.EnumField('TypeValueValuesEnum', 5)