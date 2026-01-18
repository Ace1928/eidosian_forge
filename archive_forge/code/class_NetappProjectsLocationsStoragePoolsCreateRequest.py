from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetappProjectsLocationsStoragePoolsCreateRequest(_messages.Message):
    """A NetappProjectsLocationsStoragePoolsCreateRequest object.

  Fields:
    parent: Required. Value for parent.
    storagePool: A StoragePool resource to be passed as the request body.
    storagePoolId: Required. Id of the requesting storage pool If auto-
      generating Id server-side, remove this field and id from the
      method_signature of Create RPC
  """
    parent = _messages.StringField(1, required=True)
    storagePool = _messages.MessageField('StoragePool', 2)
    storagePoolId = _messages.StringField(3)