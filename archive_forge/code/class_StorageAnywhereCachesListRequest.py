from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StorageAnywhereCachesListRequest(_messages.Message):
    """A StorageAnywhereCachesListRequest object.

  Fields:
    bucket: Name of the parent bucket.
    pageSize: Maximum number of items to return in a single page of responses.
      Maximum 1000.
    pageToken: A previously-returned page token representing part of the
      larger set of results to view.
  """
    bucket = _messages.StringField(1, required=True)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)