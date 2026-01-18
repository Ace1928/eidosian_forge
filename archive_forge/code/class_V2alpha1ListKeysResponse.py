from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class V2alpha1ListKeysResponse(_messages.Message):
    """Response message for `ListKeys` method.

  Fields:
    keys: A list of API keys.
    nextPageToken: The pagination token for the next page of results.
  """
    keys = _messages.MessageField('V2alpha1ApiKey', 1, repeated=True)
    nextPageToken = _messages.StringField(2)