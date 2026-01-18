from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListNotificationConfigsResponse(_messages.Message):
    """Response message for listing notification configs.

  Fields:
    nextPageToken: Token to retrieve the next page of results, or empty if
      there are no more results.
    notificationConfigs: Notification configs belonging to the requested
      parent.
  """
    nextPageToken = _messages.StringField(1)
    notificationConfigs = _messages.MessageField('NotificationConfig', 2, repeated=True)