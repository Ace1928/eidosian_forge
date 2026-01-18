from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionAutoMlTablesInputsTransformationNumericArrayTransformation(_messages.Message):
    """Treats the column as numerical array and performs following
  transformation functions. * All transformations for Numerical types applied
  to the average of the all elements. * The average of empty arrays is treated
  as zero.

  Fields:
    columnName: A string attribute.
    invalidValuesAllowed: If invalid values is allowed, the training pipeline
      will create a boolean feature that indicated whether the value is valid.
      Otherwise, the training pipeline will discard the input row from
      trainining data.
  """
    columnName = _messages.StringField(1)
    invalidValuesAllowed = _messages.BooleanField(2)