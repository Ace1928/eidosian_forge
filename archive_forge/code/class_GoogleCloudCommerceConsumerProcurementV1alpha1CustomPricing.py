from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudCommerceConsumerProcurementV1alpha1CustomPricing(_messages.Message):
    """Information about custom pricing on a resource.

  Fields:
    endTime: The end time of the custom pricing.
  """
    endTime = _messages.StringField(1)