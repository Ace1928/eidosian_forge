from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EventSubscription(_messages.Message):
    """represents the Connector's EventSubscription resource

  Fields:
    createTime: Output only. Created time.
    destinations: Optional. The destination to hit when we receive an event
    eventTypeId: Optional. Event type id of the event of current
      EventSubscription.
    name: Required. Resource name of the EventSubscription. Format: projects/{
      project}/locations/{location}/connections/{connection}/eventSubscription
      s/{event_subscription}
    status: Optional. Status indicates the status of the event subscription
      resource
    subscriber: Optional. name of the Subscriber for the current
      EventSubscription.
    subscriberLink: Optional. Link for Subscriber of the current
      EventSubscription.
    updateTime: Output only. Updated time.
  """
    createTime = _messages.StringField(1)
    destinations = _messages.MessageField('EventSubscriptionDestination', 2)
    eventTypeId = _messages.StringField(3)
    name = _messages.StringField(4)
    status = _messages.MessageField('EventSubscriptionStatus', 5)
    subscriber = _messages.StringField(6)
    subscriberLink = _messages.StringField(7)
    updateTime = _messages.StringField(8)