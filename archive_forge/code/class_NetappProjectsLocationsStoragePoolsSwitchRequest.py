from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetappProjectsLocationsStoragePoolsSwitchRequest(_messages.Message):
    """A NetappProjectsLocationsStoragePoolsSwitchRequest object.

  Fields:
    name: Required. Name of the storage pool
    switchActiveReplicaZoneRequest: A SwitchActiveReplicaZoneRequest resource
      to be passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    switchActiveReplicaZoneRequest = _messages.MessageField('SwitchActiveReplicaZoneRequest', 2)