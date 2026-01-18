from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListWorkforcePoolProviderKeysResponse(_messages.Message):
    """Response message for ListWorkforcePoolProviderKeys.

  Fields:
    nextPageToken: A token, which can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
    workforcePoolProviderKeys: A list of WorkforcePoolProviderKeys.
  """
    nextPageToken = _messages.StringField(1)
    workforcePoolProviderKeys = _messages.MessageField('WorkforcePoolProviderKey', 2, repeated=True)