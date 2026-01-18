from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1SchemaSchemaElement(_messages.Message):
    """Message type for the schema element

  Fields:
    name: Name of the field.
    properties: Properties for the schema field. For example: { "createTime":
      "2016-02-26T10:23:09.592Z", "custom": "false", "type": "string" }
  """
    name = _messages.StringField(1)
    properties = _messages.MessageField('GoogleCloudApigeeV1SchemaSchemaProperty', 2)