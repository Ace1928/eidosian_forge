from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SafetyInput(_messages.Message):
    """Input for safety metric.

  Fields:
    instance: Required. Safety instance.
    metricSpec: Required. Spec for safety metric.
  """
    instance = _messages.MessageField('GoogleCloudAiplatformV1beta1SafetyInstance', 1)
    metricSpec = _messages.MessageField('GoogleCloudAiplatformV1beta1SafetySpec', 2)