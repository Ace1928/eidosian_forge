from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1DataProfileSpecSelectedFields(_messages.Message):
    """The specification for fields to include or exclude in data profile scan.

  Fields:
    fieldNames: Optional. Expected input is a list of fully qualified names of
      fields as in the schema.Only top-level field names for nested fields are
      supported. For instance, if 'x' is of nested field type, listing 'x' is
      supported but 'x.y.z' is not supported. Here 'y' and 'y.z' are nested
      fields of 'x'.
  """
    fieldNames = _messages.StringField(1, repeated=True)