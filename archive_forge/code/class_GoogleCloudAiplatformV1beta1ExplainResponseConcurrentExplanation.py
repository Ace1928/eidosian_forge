from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ExplainResponseConcurrentExplanation(_messages.Message):
    """This message is a wrapper grouping Concurrent Explanations.

  Fields:
    explanations: The explanations of the Model's PredictResponse.predictions.
      It has the same number of elements as instances to be explained.
  """
    explanations = _messages.MessageField('GoogleCloudAiplatformV1beta1Explanation', 1, repeated=True)