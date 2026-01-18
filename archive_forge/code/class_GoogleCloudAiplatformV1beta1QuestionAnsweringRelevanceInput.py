from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1QuestionAnsweringRelevanceInput(_messages.Message):
    """Input for question answering relevance metric.

  Fields:
    instance: Required. Question answering relevance instance.
    metricSpec: Required. Spec for question answering relevance score metric.
  """
    instance = _messages.MessageField('GoogleCloudAiplatformV1beta1QuestionAnsweringRelevanceInstance', 1)
    metricSpec = _messages.MessageField('GoogleCloudAiplatformV1beta1QuestionAnsweringRelevanceSpec', 2)