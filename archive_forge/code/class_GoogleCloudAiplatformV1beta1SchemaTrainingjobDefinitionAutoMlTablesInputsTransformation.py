from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionAutoMlTablesInputsTransformation(_messages.Message):
    """A GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionAutoMlTablesInp
  utsTransformation object.

  Fields:
    auto: A GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionAutoMlTable
      sInputsTransformationAutoTransformation attribute.
    categorical: A GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionAuto
      MlTablesInputsTransformationCategoricalTransformation attribute.
    numeric: A GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionAutoMlTa
      blesInputsTransformationNumericTransformation attribute.
    repeatedCategorical: A GoogleCloudAiplatformV1beta1SchemaTrainingjobDefini
      tionAutoMlTablesInputsTransformationCategoricalArrayTransformation
      attribute.
    repeatedNumeric: A GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinition
      AutoMlTablesInputsTransformationNumericArrayTransformation attribute.
    repeatedText: A GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionAut
      oMlTablesInputsTransformationTextArrayTransformation attribute.
    text: A GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionAutoMlTable
      sInputsTransformationTextTransformation attribute.
    timestamp: A GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionAutoMl
      TablesInputsTransformationTimestampTransformation attribute.
  """
    auto = _messages.MessageField('GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionAutoMlTablesInputsTransformationAutoTransformation', 1)
    categorical = _messages.MessageField('GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionAutoMlTablesInputsTransformationCategoricalTransformation', 2)
    numeric = _messages.MessageField('GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionAutoMlTablesInputsTransformationNumericTransformation', 3)
    repeatedCategorical = _messages.MessageField('GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionAutoMlTablesInputsTransformationCategoricalArrayTransformation', 4)
    repeatedNumeric = _messages.MessageField('GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionAutoMlTablesInputsTransformationNumericArrayTransformation', 5)
    repeatedText = _messages.MessageField('GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionAutoMlTablesInputsTransformationTextArrayTransformation', 6)
    text = _messages.MessageField('GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionAutoMlTablesInputsTransformationTextTransformation', 7)
    timestamp = _messages.MessageField('GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionAutoMlTablesInputsTransformationTimestampTransformation', 8)