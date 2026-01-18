from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PubsubliteAdminProjectsLocationsSubscriptionsCreateRequest(_messages.Message):
    """A PubsubliteAdminProjectsLocationsSubscriptionsCreateRequest object.

  Fields:
    parent: Required. The parent location in which to create the subscription.
      Structured like `projects/{project_number}/locations/{location}`.
    skipBacklog: If true, the newly created subscription will only receive
      messages published after the subscription was created. Otherwise, the
      entire message backlog will be received on the subscription. Defaults to
      false.
    subscription: A Subscription resource to be passed as the request body.
    subscriptionId: Required. The ID to use for the subscription, which will
      become the final component of the subscription's name. This value is
      structured like: `my-sub-name`.
  """
    parent = _messages.StringField(1, required=True)
    skipBacklog = _messages.BooleanField(2)
    subscription = _messages.MessageField('Subscription', 3)
    subscriptionId = _messages.StringField(4)