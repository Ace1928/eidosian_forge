from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1SchemaTrainingjobDefinitionAutoMlForecastingInputsTransformation(_messages.Message):
    """A GoogleCloudAiplatformV1SchemaTrainingjobDefinitionAutoMlForecastingInp
  utsTransformation object.

  Fields:
    auto: A GoogleCloudAiplatformV1SchemaTrainingjobDefinitionAutoMlForecastin
      gInputsTransformationAutoTransformation attribute.
    categorical: A GoogleCloudAiplatformV1SchemaTrainingjobDefinitionAutoMlFor
      ecastingInputsTransformationCategoricalTransformation attribute.
    numeric: A GoogleCloudAiplatformV1SchemaTrainingjobDefinitionAutoMlForecas
      tingInputsTransformationNumericTransformation attribute.
    text: A GoogleCloudAiplatformV1SchemaTrainingjobDefinitionAutoMlForecastin
      gInputsTransformationTextTransformation attribute.
    timestamp: A GoogleCloudAiplatformV1SchemaTrainingjobDefinitionAutoMlForec
      astingInputsTransformationTimestampTransformation attribute.
  """
    auto = _messages.MessageField('GoogleCloudAiplatformV1SchemaTrainingjobDefinitionAutoMlForecastingInputsTransformationAutoTransformation', 1)
    categorical = _messages.MessageField('GoogleCloudAiplatformV1SchemaTrainingjobDefinitionAutoMlForecastingInputsTransformationCategoricalTransformation', 2)
    numeric = _messages.MessageField('GoogleCloudAiplatformV1SchemaTrainingjobDefinitionAutoMlForecastingInputsTransformationNumericTransformation', 3)
    text = _messages.MessageField('GoogleCloudAiplatformV1SchemaTrainingjobDefinitionAutoMlForecastingInputsTransformationTextTransformation', 4)
    timestamp = _messages.MessageField('GoogleCloudAiplatformV1SchemaTrainingjobDefinitionAutoMlForecastingInputsTransformationTimestampTransformation', 5)