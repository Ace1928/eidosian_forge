from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StorageObjectsBulkRestoreRequest(_messages.Message):
    """A StorageObjectsBulkRestoreRequest object.

  Fields:
    bucket: Name of the bucket in which the object resides.
    bulkRestoreObjectsRequest: A BulkRestoreObjectsRequest resource to be
      passed as the request body.
  """
    bucket = _messages.StringField(1, required=True)
    bulkRestoreObjectsRequest = _messages.MessageField('BulkRestoreObjectsRequest', 2)