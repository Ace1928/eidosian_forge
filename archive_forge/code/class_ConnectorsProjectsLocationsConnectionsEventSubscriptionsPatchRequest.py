from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConnectorsProjectsLocationsConnectionsEventSubscriptionsPatchRequest(_messages.Message):
    """A ConnectorsProjectsLocationsConnectionsEventSubscriptionsPatchRequest
  object.

  Fields:
    eventSubscription: A EventSubscription resource to be passed as the
      request body.
    name: Required. Resource name of the EventSubscription. Format: projects/{
      project}/locations/{location}/connections/{connection}/eventSubscription
      s/{event_subscription}
    updateMask: Required. The list of fields to update. Fields are specified
      relative to the Subscription. A field will be overwritten if it is in
      the mask. You can modify only the fields listed below. To update the
      EventSubscription details: * `serviceAccount`
  """
    eventSubscription = _messages.MessageField('EventSubscription', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)