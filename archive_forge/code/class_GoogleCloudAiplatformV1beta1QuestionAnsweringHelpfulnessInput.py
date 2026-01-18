from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1QuestionAnsweringHelpfulnessInput(_messages.Message):
    """Input for question answering helpfulness metric.

  Fields:
    instance: Required. Question answering helpfulness instance.
    metricSpec: Required. Spec for question answering helpfulness score
      metric.
  """
    instance = _messages.MessageField('GoogleCloudAiplatformV1beta1QuestionAnsweringHelpfulnessInstance', 1)
    metricSpec = _messages.MessageField('GoogleCloudAiplatformV1beta1QuestionAnsweringHelpfulnessSpec', 2)