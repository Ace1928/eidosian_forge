from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetappProjectsLocationsVolumesCreateRequest(_messages.Message):
    """A NetappProjectsLocationsVolumesCreateRequest object.

  Fields:
    parent: Required. Value for parent.
    volume: A Volume resource to be passed as the request body.
    volumeId: Required. Id of the requesting volume If auto-generating Id
      server-side, remove this field and Id from the method_signature of
      Create RPC
  """
    parent = _messages.StringField(1, required=True)
    volume = _messages.MessageField('Volume', 2)
    volumeId = _messages.StringField(3)