from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkPlacement(_messages.Message):
    """NetworkPlacement Represents a Google managed network placement resource.

  Fields:
    creationTimestamp: [Output Only] Creation timestamp in RFC3339 text
      format.
    description: [Output Only] An optional description of this resource.
    features: [Output Only] Features supported by the network.
    id: [Output Only] The unique identifier for the resource. This identifier
      is defined by the server.
    kind: [Output Only] Type of the resource. Always compute#networkPlacement
      for network placements.
    name: [Output Only] Name of the resource.
    selfLink: [Output Only] Server-defined URL for the resource.
    selfLinkWithId: [Output Only] Server-defined URL for this resource with
      the resource id.
    zone: [Output Only] Zone to which the network is restricted.
  """
    creationTimestamp = _messages.StringField(1)
    description = _messages.StringField(2)
    features = _messages.MessageField('NetworkPlacementNetworkFeatures', 3)
    id = _messages.IntegerField(4, variant=_messages.Variant.UINT64)
    kind = _messages.StringField(5, default='compute#networkPlacement')
    name = _messages.StringField(6)
    selfLink = _messages.StringField(7)
    selfLinkWithId = _messages.StringField(8)
    zone = _messages.StringField(9)