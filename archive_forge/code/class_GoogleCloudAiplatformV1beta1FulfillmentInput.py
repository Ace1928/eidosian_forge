from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1FulfillmentInput(_messages.Message):
    """Input for fulfillment metric.

  Fields:
    instance: Required. Fulfillment instance.
    metricSpec: Required. Spec for fulfillment score metric.
  """
    instance = _messages.MessageField('GoogleCloudAiplatformV1beta1FulfillmentInstance', 1)
    metricSpec = _messages.MessageField('GoogleCloudAiplatformV1beta1FulfillmentSpec', 2)