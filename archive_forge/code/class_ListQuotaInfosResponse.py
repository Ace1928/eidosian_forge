from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ListQuotaInfosResponse(_messages.Message):
    """Message for response to listing QuotaInfos

  Fields:
    nextPageToken: A token, which can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
    quotaInfos: The list of QuotaInfo
  """
    nextPageToken = _messages.StringField(1)
    quotaInfos = _messages.MessageField('QuotaInfo', 2, repeated=True)