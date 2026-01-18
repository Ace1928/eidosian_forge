from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
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