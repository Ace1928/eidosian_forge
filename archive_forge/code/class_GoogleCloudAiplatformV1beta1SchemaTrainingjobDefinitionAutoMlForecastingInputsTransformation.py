from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionAutoMlForecastingInputsTransformation(_messages.Message):
    """A GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionAutoMlForecasti
  ngInputsTransformation object.

  Fields:
    auto: A GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionAutoMlForec
      astingInputsTransformationAutoTransformation attribute.
    categorical: A GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionAuto
      MlForecastingInputsTransformationCategoricalTransformation attribute.
    numeric: A GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionAutoMlFo
      recastingInputsTransformationNumericTransformation attribute.
    text: A GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionAutoMlForec
      astingInputsTransformationTextTransformation attribute.
    timestamp: A GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionAutoMl
      ForecastingInputsTransformationTimestampTransformation attribute.
  """
    auto = _messages.MessageField('GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionAutoMlForecastingInputsTransformationAutoTransformation', 1)
    categorical = _messages.MessageField('GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionAutoMlForecastingInputsTransformationCategoricalTransformation', 2)
    numeric = _messages.MessageField('GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionAutoMlForecastingInputsTransformationNumericTransformation', 3)
    text = _messages.MessageField('GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionAutoMlForecastingInputsTransformationTextTransformation', 4)
    timestamp = _messages.MessageField('GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionAutoMlForecastingInputsTransformationTimestampTransformation', 5)