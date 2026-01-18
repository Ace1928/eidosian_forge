from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1SchemaTrainingjobDefinitionAutoMlTablesInputsTransformation(_messages.Message):
    """A GoogleCloudAiplatformV1SchemaTrainingjobDefinitionAutoMlTablesInputsTr
  ansformation object.

  Fields:
    auto: A GoogleCloudAiplatformV1SchemaTrainingjobDefinitionAutoMlTablesInpu
      tsTransformationAutoTransformation attribute.
    categorical: A GoogleCloudAiplatformV1SchemaTrainingjobDefinitionAutoMlTab
      lesInputsTransformationCategoricalTransformation attribute.
    numeric: A GoogleCloudAiplatformV1SchemaTrainingjobDefinitionAutoMlTablesI
      nputsTransformationNumericTransformation attribute.
    repeatedCategorical: A GoogleCloudAiplatformV1SchemaTrainingjobDefinitionA
      utoMlTablesInputsTransformationCategoricalArrayTransformation attribute.
    repeatedNumeric: A GoogleCloudAiplatformV1SchemaTrainingjobDefinitionAutoM
      lTablesInputsTransformationNumericArrayTransformation attribute.
    repeatedText: A GoogleCloudAiplatformV1SchemaTrainingjobDefinitionAutoMlTa
      blesInputsTransformationTextArrayTransformation attribute.
    text: A GoogleCloudAiplatformV1SchemaTrainingjobDefinitionAutoMlTablesInpu
      tsTransformationTextTransformation attribute.
    timestamp: A GoogleCloudAiplatformV1SchemaTrainingjobDefinitionAutoMlTable
      sInputsTransformationTimestampTransformation attribute.
  """
    auto = _messages.MessageField('GoogleCloudAiplatformV1SchemaTrainingjobDefinitionAutoMlTablesInputsTransformationAutoTransformation', 1)
    categorical = _messages.MessageField('GoogleCloudAiplatformV1SchemaTrainingjobDefinitionAutoMlTablesInputsTransformationCategoricalTransformation', 2)
    numeric = _messages.MessageField('GoogleCloudAiplatformV1SchemaTrainingjobDefinitionAutoMlTablesInputsTransformationNumericTransformation', 3)
    repeatedCategorical = _messages.MessageField('GoogleCloudAiplatformV1SchemaTrainingjobDefinitionAutoMlTablesInputsTransformationCategoricalArrayTransformation', 4)
    repeatedNumeric = _messages.MessageField('GoogleCloudAiplatformV1SchemaTrainingjobDefinitionAutoMlTablesInputsTransformationNumericArrayTransformation', 5)
    repeatedText = _messages.MessageField('GoogleCloudAiplatformV1SchemaTrainingjobDefinitionAutoMlTablesInputsTransformationTextArrayTransformation', 6)
    text = _messages.MessageField('GoogleCloudAiplatformV1SchemaTrainingjobDefinitionAutoMlTablesInputsTransformationTextTransformation', 7)
    timestamp = _messages.MessageField('GoogleCloudAiplatformV1SchemaTrainingjobDefinitionAutoMlTablesInputsTransformationTimestampTransformation', 8)