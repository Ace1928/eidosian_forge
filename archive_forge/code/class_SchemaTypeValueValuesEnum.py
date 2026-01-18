from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
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