from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SchemaFileFormatValueValuesEnum(_messages.Enum):
    """The schema file format along JSON data files.

    Values:
      SCHEMA_FILE_FORMAT_UNSPECIFIED: Unspecified schema file format.
      NO_SCHEMA_FILE: Do not attach schema file.
      AVRO_SCHEMA_FILE: Avro schema format.
    """
    SCHEMA_FILE_FORMAT_UNSPECIFIED = 0
    NO_SCHEMA_FILE = 1
    AVRO_SCHEMA_FILE = 2