from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1ActionIncompatibleDataSchema(_messages.Message):
    """Action details for incompatible schemas detected by discovery.

  Enums:
    SchemaChangeValueValuesEnum: Whether the action relates to a schema that
      is incompatible or modified.

  Fields:
    existingSchema: The existing and expected schema of the table. The schema
      is provided as a JSON formatted structure listing columns and data
      types.
    newSchema: The new and incompatible schema within the table. The schema is
      provided as a JSON formatted structured listing columns and data types.
    sampledDataLocations: The list of data locations sampled and used for
      format/schema inference.
    schemaChange: Whether the action relates to a schema that is incompatible
      or modified.
    table: The name of the table containing invalid data.
  """

    class SchemaChangeValueValuesEnum(_messages.Enum):
        """Whether the action relates to a schema that is incompatible or
    modified.

    Values:
      SCHEMA_CHANGE_UNSPECIFIED: Schema change unspecified.
      INCOMPATIBLE: Newly discovered schema is incompatible with existing
        schema.
      MODIFIED: Newly discovered schema has changed from existing schema for
        data in a curated zone.
    """
        SCHEMA_CHANGE_UNSPECIFIED = 0
        INCOMPATIBLE = 1
        MODIFIED = 2
    existingSchema = _messages.StringField(1)
    newSchema = _messages.StringField(2)
    sampledDataLocations = _messages.StringField(3, repeated=True)
    schemaChange = _messages.EnumField('SchemaChangeValueValuesEnum', 4)
    table = _messages.StringField(5)