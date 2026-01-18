from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1DocumentSchemaEntityTypeEnumValues(_messages.Message):
    """Defines the a list of enum values.

  Fields:
    values: The individual values that this enum values type can include.
  """
    values = _messages.StringField(1, repeated=True)