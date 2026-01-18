from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1SchemaTrainingjobDefinitionAutoMlTablesInputsTransformationTextArrayTransformation(_messages.Message):
    """Treats the column as text array and performs following transformation
  functions. * Concatenate all text values in the array into a single text
  value using a space (" ") as a delimiter, and then treat the result as a
  single text value. Apply the transformations for Text columns. * Empty
  arrays treated as an empty text.

  Fields:
    columnName: A string attribute.
  """
    columnName = _messages.StringField(1)