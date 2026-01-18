from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1PairwiseSummarizationQualityInput(_messages.Message):
    """Input for pairwise summarization quality metric.

  Fields:
    instance: Required. Pairwise summarization quality instance.
    metricSpec: Required. Spec for pairwise summarization quality score
      metric.
  """
    instance = _messages.MessageField('GoogleCloudAiplatformV1beta1PairwiseSummarizationQualityInstance', 1)
    metricSpec = _messages.MessageField('GoogleCloudAiplatformV1beta1PairwiseSummarizationQualitySpec', 2)