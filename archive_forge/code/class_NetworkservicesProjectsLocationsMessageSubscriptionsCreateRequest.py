from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsMessageSubscriptionsCreateRequest(_messages.Message):
    """A NetworkservicesProjectsLocationsMessageSubscriptionsCreateRequest
  object.

  Fields:
    messageSubscription: A MessageSubscription resource to be passed as the
      request body.
    messageSubscriptionId: Required. Short name of the MessageSubscription
      resource to be created.
    parent: Required. The parent resource of the MessageSubscription. Must be
      in the format `projects/*/locations/*`.
  """
    messageSubscription = _messages.MessageField('MessageSubscription', 1)
    messageSubscriptionId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)