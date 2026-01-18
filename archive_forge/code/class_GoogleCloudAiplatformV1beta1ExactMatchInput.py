from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ExactMatchInput(_messages.Message):
    """Input for exact match metric.

  Fields:
    instances: Required. Repeated exact match instances.
    metricSpec: Required. Spec for exact match metric.
  """
    instances = _messages.MessageField('GoogleCloudAiplatformV1beta1ExactMatchInstance', 1, repeated=True)
    metricSpec = _messages.MessageField('GoogleCloudAiplatformV1beta1ExactMatchSpec', 2)