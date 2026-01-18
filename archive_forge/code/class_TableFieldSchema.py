from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TableFieldSchema(_messages.Message):
    """A TableFieldSchema object.

  Fields:
    description: [Optional] The field description. The maximum length is 16K
      characters.
    fields: [Optional] Describes the nested schema fields if the type property
      is set to RECORD.
    mode: [Optional] The field mode. Possible values include NULLABLE,
      REQUIRED and REPEATED. The default value is NULLABLE.
    name: [Required] The field name. The name must contain only letters (a-z,
      A-Z), numbers (0-9), or underscores (_), and must start with a letter or
      underscore. The maximum length is 128 characters.
    type: [Required] The field data type. Possible values include STRING,
      BYTES, INTEGER, FLOAT, BOOLEAN, TIMESTAMP or RECORD (where RECORD
      indicates that the field contains a nested schema).
  """
    description = _messages.StringField(1)
    fields = _messages.MessageField('TableFieldSchema', 2, repeated=True)
    mode = _messages.StringField(3)
    name = _messages.StringField(4)
    type = _messages.StringField(5)