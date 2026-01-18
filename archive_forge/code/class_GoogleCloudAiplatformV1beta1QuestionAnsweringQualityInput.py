from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1QuestionAnsweringQualityInput(_messages.Message):
    """Input for question answering quality metric.

  Fields:
    instance: Required. Question answering quality instance.
    metricSpec: Required. Spec for question answering quality score metric.
  """
    instance = _messages.MessageField('GoogleCloudAiplatformV1beta1QuestionAnsweringQualityInstance', 1)
    metricSpec = _messages.MessageField('GoogleCloudAiplatformV1beta1QuestionAnsweringQualitySpec', 2)