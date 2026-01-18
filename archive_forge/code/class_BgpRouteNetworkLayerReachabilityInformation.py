from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BgpRouteNetworkLayerReachabilityInformation(_messages.Message):
    """Network Layer Reachability Information (NLRI) for a route.

  Fields:
    pathId: If the BGP session supports multiple paths (RFC 7911), the path
      identifier for this route.
    prefix: Human readable CIDR notation for a prefix. E.g. 10.42.0.0/16.
  """
    pathId = _messages.IntegerField(1, variant=_messages.Variant.UINT32)
    prefix = _messages.StringField(2)