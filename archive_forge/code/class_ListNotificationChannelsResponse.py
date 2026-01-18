from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListNotificationChannelsResponse(_messages.Message):
    """The ListNotificationChannels response.

  Fields:
    nextPageToken: If not empty, indicates that there may be more results that
      match the request. Use the value in the page_token field in a subsequent
      request to fetch the next set of results. If empty, all results have
      been returned.
    notificationChannels: The notification channels defined for the specified
      project.
    totalSize: The total number of notification channels in all pages. This
      number is only an estimate, and may change in subsequent pages.
      https://aip.dev/158
  """
    nextPageToken = _messages.StringField(1)
    notificationChannels = _messages.MessageField('NotificationChannel', 2, repeated=True)
    totalSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)