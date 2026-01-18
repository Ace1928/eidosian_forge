from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1MetadataSchema(_messages.Message):
    """Instance of a general MetadataSchema.

  Enums:
    SchemaTypeValueValuesEnum: The type of the MetadataSchema. This is a
      property that identifies which metadata types will use the
      MetadataSchema.

  Fields:
    createTime: Output only. Timestamp when this MetadataSchema was created.
    description: Description of the Metadata Schema
    name: Output only. The resource name of the MetadataSchema.
    schema: Required. The raw YAML string representation of the
      MetadataSchema. The combination of [MetadataSchema.version] and the
      schema name given by `title` in [MetadataSchema.schema] must be unique
      within a MetadataStore. The schema is defined as an OpenAPI 3.0.2
      [MetadataSchema Object](https://github.com/OAI/OpenAPI-
      Specification/blob/master/versions/3.0.2.md#schemaObject)
    schemaType: The type of the MetadataSchema. This is a property that
      identifies which metadata types will use the MetadataSchema.
    schemaVersion: The version of the MetadataSchema. The version's format
      must match the following regular expression: `^[0-9]+.+.+$`, which would
      allow to order/compare different versions. Example: 1.0.0, 1.0.1, etc.
  """

    class SchemaTypeValueValuesEnum(_messages.Enum):
        """The type of the MetadataSchema. This is a property that identifies
    which metadata types will use the MetadataSchema.

    Values:
      METADATA_SCHEMA_TYPE_UNSPECIFIED: Unspecified type for the
        MetadataSchema.
      ARTIFACT_TYPE: A type indicating that the MetadataSchema will be used by
        Artifacts.
      EXECUTION_TYPE: A typee indicating that the MetadataSchema will be used
        by Executions.
      CONTEXT_TYPE: A state indicating that the MetadataSchema will be used by
        Contexts.
    """
        METADATA_SCHEMA_TYPE_UNSPECIFIED = 0
        ARTIFACT_TYPE = 1
        EXECUTION_TYPE = 2
        CONTEXT_TYPE = 3
    createTime = _messages.StringField(1)
    description = _messages.StringField(2)
    name = _messages.StringField(3)
    schema = _messages.StringField(4)
    schemaType = _messages.EnumField('SchemaTypeValueValuesEnum', 5)
    schemaVersion = _messages.StringField(6)