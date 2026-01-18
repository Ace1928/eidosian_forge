from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SchemaSettings(_messages.Message):
    """Settings for validating messages published against a schema.

  Enums:
    EncodingValueValuesEnum: Optional. The encoding of messages validated
      against `schema`.

  Fields:
    encoding: Optional. The encoding of messages validated against `schema`.
    firstRevisionId: Optional. The minimum (inclusive) revision allowed for
      validating messages. If empty or not present, allow any revision to be
      validated against last_revision or any revision created before.
    lastRevisionId: Optional. The maximum (inclusive) revision allowed for
      validating messages. If empty or not present, allow any revision to be
      validated against first_revision or any revision created after.
    schema: Required. The name of the schema that messages published should be
      validated against. Format is `projects/{project}/schemas/{schema}`. The
      value of this field will be `_deleted-schema_` if the schema has been
      deleted.
  """

    class EncodingValueValuesEnum(_messages.Enum):
        """Optional. The encoding of messages validated against `schema`.

    Values:
      ENCODING_UNSPECIFIED: Unspecified
      JSON: JSON encoding
      BINARY: Binary encoding, as defined by the schema type. For some schema
        types, binary encoding may not be available.
    """
        ENCODING_UNSPECIFIED = 0
        JSON = 1
        BINARY = 2
    encoding = _messages.EnumField('EncodingValueValuesEnum', 1)
    firstRevisionId = _messages.StringField(2)
    lastRevisionId = _messages.StringField(3)
    schema = _messages.StringField(4)