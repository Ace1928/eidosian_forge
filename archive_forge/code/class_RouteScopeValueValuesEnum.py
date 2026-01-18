from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RouteScopeValueValuesEnum(_messages.Enum):
    """Indicates where route is applicable.

    Values:
      ROUTE_SCOPE_UNSPECIFIED: Unspecified scope. Default value.
      NETWORK: Route is applicable to packets in Network.
      NCC_HUB: Route is applicable to packets using NCC Hub's routing table.
    """
    ROUTE_SCOPE_UNSPECIFIED = 0
    NETWORK = 1
    NCC_HUB = 2