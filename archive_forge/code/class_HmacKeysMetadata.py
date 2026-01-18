from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HmacKeysMetadata(_messages.Message):
    """A list of hmacKeys.

  Fields:
    items: The list of items.
    kind: The kind of item this is. For lists of hmacKeys, this is always
      storage#hmacKeysMetadata.
    nextPageToken: The continuation token, used to page through large result
      sets. Provide this value in a subsequent request to return the next page
      of results.
  """
    items = _messages.MessageField('HmacKeyMetadata', 1, repeated=True)
    kind = _messages.StringField(2, default='storage#hmacKeysMetadata')
    nextPageToken = _messages.StringField(3)