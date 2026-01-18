from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListInstanceConfigOperationsResponse(_messages.Message):
    """The response for ListInstanceConfigOperations.

  Fields:
    nextPageToken: `next_page_token` can be sent in a subsequent
      ListInstanceConfigOperations call to fetch more of the matching
      metadata.
    operations: The list of matching instance config long-running operations.
      Each operation's name will be prefixed by the instance config's name.
      The operation's metadata field type `metadata.type_url` describes the
      type of the metadata.
  """
    nextPageToken = _messages.StringField(1)
    operations = _messages.MessageField('Operation', 2, repeated=True)