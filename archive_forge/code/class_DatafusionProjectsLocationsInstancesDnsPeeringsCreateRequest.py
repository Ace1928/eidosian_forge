from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatafusionProjectsLocationsInstancesDnsPeeringsCreateRequest(_messages.Message):
    """A DatafusionProjectsLocationsInstancesDnsPeeringsCreateRequest object.

  Fields:
    dnsPeering: A DnsPeering resource to be passed as the request body.
    dnsPeeringId: Required. The name of the peering to create.
    parent: Required. The resource on which DNS peering will be created.
  """
    dnsPeering = _messages.MessageField('DnsPeering', 1)
    dnsPeeringId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)