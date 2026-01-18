from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionTftForecastingInputsTransformation(_messages.Message):
    """A GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionTftForecastingI
  nputsTransformation object.

  Fields:
    auto: A GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionTftForecast
      ingInputsTransformationAutoTransformation attribute.
    categorical: A GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionTftF
      orecastingInputsTransformationCategoricalTransformation attribute.
    numeric: A GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionTftForec
      astingInputsTransformationNumericTransformation attribute.
    text: A GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionTftForecast
      ingInputsTransformationTextTransformation attribute.
    timestamp: A GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionTftFor
      ecastingInputsTransformationTimestampTransformation attribute.
  """
    auto = _messages.MessageField('GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionTftForecastingInputsTransformationAutoTransformation', 1)
    categorical = _messages.MessageField('GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionTftForecastingInputsTransformationCategoricalTransformation', 2)
    numeric = _messages.MessageField('GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionTftForecastingInputsTransformationNumericTransformation', 3)
    text = _messages.MessageField('GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionTftForecastingInputsTransformationTextTransformation', 4)
    timestamp = _messages.MessageField('GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionTftForecastingInputsTransformationTimestampTransformation', 5)