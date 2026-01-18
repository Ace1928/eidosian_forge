from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1ExplainResponse(_messages.Message):
    """Response message for PredictionService.Explain.

  Fields:
    deployedModelId: ID of the Endpoint's DeployedModel that served this
      explanation.
    explanations: The explanations of the Model's PredictResponse.predictions.
      It has the same number of elements as instances to be explained.
    predictions: The predictions that are the output of the predictions call.
      Same as PredictResponse.predictions.
  """
    deployedModelId = _messages.StringField(1)
    explanations = _messages.MessageField('GoogleCloudAiplatformV1Explanation', 2, repeated=True)
    predictions = _messages.MessageField('extra_types.JsonValue', 3, repeated=True)