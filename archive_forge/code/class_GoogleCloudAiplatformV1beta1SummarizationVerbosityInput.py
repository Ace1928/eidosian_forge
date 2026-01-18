from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SummarizationVerbosityInput(_messages.Message):
    """Input for summarization verbosity metric.

  Fields:
    instance: Required. Summarization verbosity instance.
    metricSpec: Required. Spec for summarization verbosity score metric.
  """
    instance = _messages.MessageField('GoogleCloudAiplatformV1beta1SummarizationVerbosityInstance', 1)
    metricSpec = _messages.MessageField('GoogleCloudAiplatformV1beta1SummarizationVerbositySpec', 2)