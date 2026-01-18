from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class UpdateSubscriptionRequest(_messages.Message):
    """Request for the UpdateSubscription method.

  Fields:
    subscription: Required. The updated subscription object.
    updateMask: Required. Indicates which fields in the provided subscription
      to update. Must be specified and non-empty.
  """
    subscription = _messages.MessageField('Subscription', 1)
    updateMask = _messages.StringField(2)