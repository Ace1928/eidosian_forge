from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConnectorsProjectsLocationsConnectionsEventSubscriptionsCreateRequest(_messages.Message):
    """A ConnectorsProjectsLocationsConnectionsEventSubscriptionsCreateRequest
  object.

  Fields:
    eventSubscription: A EventSubscription resource to be passed as the
      request body.
    eventSubscriptionId: Required. Identifier to assign to the Event
      Subscription. Must be unique within scope of the parent resource.
    parent: Required. Parent resource of the EventSubscription, of the form:
      `projects/*/locations/*/connections/*`
  """
    eventSubscription = _messages.MessageField('EventSubscription', 1)
    eventSubscriptionId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)