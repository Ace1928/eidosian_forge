from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PostgreSql(_messages.Message):
    """Settings for PostgreSQL data source.

  Enums:
    SchemaValidationValueValuesEnum: Optional. Configure how much Postgresql
      schema validation to perform. Default to `STRICT` if not specified.

  Fields:
    cloudSql: Cloud SQL configurations.
    database: Required. Name of the PostgreSQL database.
    schemaValidation: Optional. Configure how much Postgresql schema
      validation to perform. Default to `STRICT` if not specified.
  """

    class SchemaValidationValueValuesEnum(_messages.Enum):
        """Optional. Configure how much Postgresql schema validation to perform.
    Default to `STRICT` if not specified.

    Values:
      SQL_SCHEMA_VALIDATION_UNSPECIFIED: Unspecified SQL schema validation.
        Default to STRICT.
      NONE: Skip no SQL schema validation. Use it with extreme caution.
        CreateSchema or UpdateSchema will succeed even if SQL database is
        unavailable or SQL schema is incompatible. Generated SQL may fail at
        execution time.
      STRICT: Connect to the SQL database and validate that the SQL DDL
        matches the schema exactly. Surface any discrepancies as
        `FAILED_PRECONDITION` with an `IncompatibleSqlSchemaError` error
        detail.
    """
        SQL_SCHEMA_VALIDATION_UNSPECIFIED = 0
        NONE = 1
        STRICT = 2
    cloudSql = _messages.MessageField('CloudSqlInstance', 1)
    database = _messages.StringField(2)
    schemaValidation = _messages.EnumField('SchemaValidationValueValuesEnum', 3)