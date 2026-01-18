from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1SchemaSchemaProperty(_messages.Message):
    """Properties for the schema field.

  Fields:
    createTime: Time the field was created in RFC3339 string form. For
      example: `2016-02-26T10:23:09.592Z`.
    custom: Flag that specifies whether the field is standard in the dataset
      or a custom field created by the customer. `true` indicates that it is a
      custom field.
    type: Data type of the field.
  """
    createTime = _messages.StringField(1)
    custom = _messages.StringField(2)
    type = _messages.StringField(3)