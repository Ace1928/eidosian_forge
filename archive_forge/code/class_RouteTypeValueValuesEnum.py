from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RouteTypeValueValuesEnum(_messages.Enum):
    """Type of route.

    Values:
      ROUTE_TYPE_UNSPECIFIED: Unspecified type. Default value.
      SUBNET: Route is a subnet route automatically created by the system.
      STATIC: Static route created by the user, including the default route to
        the internet.
      DYNAMIC: Dynamic route exchanged between BGP peers.
      PEERING_SUBNET: A subnet route received from peering network.
      PEERING_STATIC: A static route received from peering network.
      PEERING_DYNAMIC: A dynamic route received from peering network.
      POLICY_BASED: Policy based route.
    """
    ROUTE_TYPE_UNSPECIFIED = 0
    SUBNET = 1
    STATIC = 2
    DYNAMIC = 3
    PEERING_SUBNET = 4
    PEERING_STATIC = 5
    PEERING_DYNAMIC = 6
    POLICY_BASED = 7