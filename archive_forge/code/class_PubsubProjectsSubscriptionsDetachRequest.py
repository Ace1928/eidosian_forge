from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PubsubProjectsSubscriptionsDetachRequest(_messages.Message):
    """A PubsubProjectsSubscriptionsDetachRequest object.

  Fields:
    subscription: Required. The subscription to detach. Format is
      `projects/{project}/subscriptions/{subscription}`.
  """
    subscription = _messages.StringField(1, required=True)