from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StorageManagedFoldersListRequest(_messages.Message):
    """A StorageManagedFoldersListRequest object.

  Fields:
    bucket: Name of the bucket containing the managed folder.
    pageSize: Maximum number of items to return in a single page of responses.
    pageToken: A previously-returned page token representing part of the
      larger set of results to view.
    prefix: The managed folder name/path prefix to filter the output list of
      results.
  """
    bucket = _messages.StringField(1, required=True)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    prefix = _messages.StringField(4)