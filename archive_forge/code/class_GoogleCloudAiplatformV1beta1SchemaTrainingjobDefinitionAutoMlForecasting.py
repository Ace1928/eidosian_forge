from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionAutoMlForecasting(_messages.Message):
    """A TrainingJob that trains and uploads an AutoML Forecasting Model.

  Fields:
    inputs: The input parameters of this TrainingJob.
    metadata: The metadata information.
  """
    inputs = _messages.MessageField('GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionAutoMlForecastingInputs', 1)
    metadata = _messages.MessageField('GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionAutoMlForecastingMetadata', 2)