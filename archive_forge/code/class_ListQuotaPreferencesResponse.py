from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ListQuotaPreferencesResponse(_messages.Message):
    """Message for response to listing QuotaPreferences

  Fields:
    nextPageToken: A token, which can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
    quotaPreferences: The list of QuotaPreference
    unreachable: Locations that could not be reached.
  """
    nextPageToken = _messages.StringField(1)
    quotaPreferences = _messages.MessageField('QuotaPreference', 2, repeated=True)
    unreachable = _messages.StringField(3, repeated=True)