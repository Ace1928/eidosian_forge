from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SummarizationQualitySpec(_messages.Message):
    """Spec for summarization quality score metric.

  Fields:
    useReference: Optional. Whether to use instance.reference to compute
      summarization quality.
    version: Optional. Which version to use for evaluation.
  """
    useReference = _messages.BooleanField(1)
    version = _messages.IntegerField(2, variant=_messages.Variant.INT32)