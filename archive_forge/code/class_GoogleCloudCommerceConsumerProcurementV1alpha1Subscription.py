from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudCommerceConsumerProcurementV1alpha1Subscription(_messages.Message):
    """Subscription information.

  Fields:
    autoRenewalEnabled: Whether auto renewal is enabled by user choice on
      current subscription. This field indicates order/subscription status
      after pending plan change is cancelled or rejected.
    endTime: The timestamp when the subscription ends, if applicable.
    startTime: The timestamp when the subscription begins, if applicable.
  """
    autoRenewalEnabled = _messages.BooleanField(1)
    endTime = _messages.StringField(2)
    startTime = _messages.StringField(3)