from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class KmsinventoryProjectsCryptoKeysListRequest(_messages.Message):
    """A KmsinventoryProjectsCryptoKeysListRequest object.

  Fields:
    pageSize: Optional. The maximum number of keys to return. The service may
      return fewer than this value. If unspecified, at most 1000 keys will be
      returned. The maximum value is 1000; values above 1000 will be coerced
      to 1000.
    pageToken: Optional. Pass this into a subsequent request in order to
      receive the next page of results.
    parent: Required. The Google Cloud project for which to retrieve key
      metadata, in the format `projects/*`
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)