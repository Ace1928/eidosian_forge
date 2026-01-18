from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PubsubliteAdminProjectsLocationsSubscriptionsPatchRequest(_messages.Message):
    """A PubsubliteAdminProjectsLocationsSubscriptionsPatchRequest object.

  Fields:
    name: The name of the subscription. Structured like: projects/{project_num
      ber}/locations/{location}/subscriptions/{subscription_id}
    subscription: A Subscription resource to be passed as the request body.
    updateMask: Required. A mask specifying the subscription fields to change.
  """
    name = _messages.StringField(1, required=True)
    subscription = _messages.MessageField('Subscription', 2)
    updateMask = _messages.StringField(3)