from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1PairwiseSummarizationQualityInstance(_messages.Message):
    """Spec for pairwise summarization quality instance.

  Fields:
    baselinePrediction: Required. Output of the baseline model.
    context: Required. Text to be summarized.
    instruction: Required. Summarization prompt for LLM.
    prediction: Required. Output of the candidate model.
    reference: Optional. Ground truth used to compare against the prediction.
  """
    baselinePrediction = _messages.StringField(1)
    context = _messages.StringField(2)
    instruction = _messages.StringField(3)
    prediction = _messages.StringField(4)
    reference = _messages.StringField(5)