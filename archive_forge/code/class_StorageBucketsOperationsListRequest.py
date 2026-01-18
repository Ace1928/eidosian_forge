from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StorageBucketsOperationsListRequest(_messages.Message):
    """A StorageBucketsOperationsListRequest object.

  Fields:
    bucket: Name of the bucket in which to look for operations.
    filter: A filter to narrow down results to a preferred subset. The
      filtering language is documented in more detail in
      [AIP-160](https://google.aip.dev/160).
    pageSize: Maximum number of items to return in a single page of responses.
      Fewer total results may be returned than requested. The service uses
      this parameter or 100 items, whichever is smaller.
    pageToken: A previously-returned page token representing part of the
      larger set of results to view.
  """
    bucket = _messages.StringField(1, required=True)
    filter = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)