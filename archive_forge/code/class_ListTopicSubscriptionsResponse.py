from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ListTopicSubscriptionsResponse(_messages.Message):
    """Response for the `ListTopicSubscriptions` method.

  Fields:
    nextPageToken: If not empty, indicates that there may be more
      subscriptions that match the request; this value should be passed in a
      new `ListTopicSubscriptionsRequest` to get more subscriptions.
    subscriptions: The names of the subscriptions that match the request.
  """
    nextPageToken = _messages.StringField(1)
    subscriptions = _messages.StringField(2, repeated=True)