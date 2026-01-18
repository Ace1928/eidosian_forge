from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1QuestionAnsweringCorrectnessInput(_messages.Message):
    """Input for question answering correctness metric.

  Fields:
    instance: Required. Question answering correctness instance.
    metricSpec: Required. Spec for question answering correctness score
      metric.
  """
    instance = _messages.MessageField('GoogleCloudAiplatformV1beta1QuestionAnsweringCorrectnessInstance', 1)
    metricSpec = _messages.MessageField('GoogleCloudAiplatformV1beta1QuestionAnsweringCorrectnessSpec', 2)