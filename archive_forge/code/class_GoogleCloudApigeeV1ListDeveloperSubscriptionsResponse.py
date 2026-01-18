from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1ListDeveloperSubscriptionsResponse(_messages.Message):
    """Response for ListDeveloperSubscriptions.

  Fields:
    developerSubscriptions: List of all subscriptions.
    nextStartKey: Value that can be sent as `startKey` to retrieve the next
      page of content. If this field is omitted, there are no subsequent
      pages.
  """
    developerSubscriptions = _messages.MessageField('GoogleCloudApigeeV1DeveloperSubscription', 1, repeated=True)
    nextStartKey = _messages.StringField(2)