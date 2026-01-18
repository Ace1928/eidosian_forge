from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MigrationJobVerificationError(_messages.Message):
    """Error message of a verification Migration job.

  Enums:
    ErrorCodeValueValuesEnum: Output only. An instance of ErrorCode specifying
      the error that occurred.

  Fields:
    errorCode: Output only. An instance of ErrorCode specifying the error that
      occurred.
    errorDetailMessage: Output only. A specific detailed error message, if
      supplied by the engine.
    errorMessage: Output only. A formatted message with further details about
      the error and a CTA.
  """

    class ErrorCodeValueValuesEnum(_messages.Enum):
        """Output only. An instance of ErrorCode specifying the error that
    occurred.

    Values:
      ERROR_CODE_UNSPECIFIED: An unknown error occurred
      CONNECTION_FAILURE: We failed to connect to one of the connection
        profile.
      AUTHENTICATION_FAILURE: We failed to authenticate to one of the
        connection profile.
      INVALID_CONNECTION_PROFILE_CONFIG: One of the involved connection
        profiles has an invalid configuration.
      VERSION_INCOMPATIBILITY: The versions of the source and the destination
        are incompatible.
      CONNECTION_PROFILE_TYPES_INCOMPATIBILITY: The types of the source and
        the destination are incompatible.
      UNSUPPORTED_GTID_MODE: The gtid_mode is not supported, applicable for
        MySQL.
      UNSUPPORTED_DEFINER: The definer is not supported.
      CANT_RESTART_RUNNING_MIGRATION: Migration is already running at the time
        of restart request.
      TABLES_WITH_LIMITED_SUPPORT: The source has tables with limited support.
        E.g. PostgreSQL tables without primary keys.
      UNSUPPORTED_DATABASE_LOCALE: The source uses an unsupported locale.
      UNSUPPORTED_DATABASE_FDW_CONFIG: The source uses an unsupported Foreign
        Data Wrapper configuration.
      ERROR_RDBMS: There was an underlying RDBMS error.
      SOURCE_SIZE_EXCEEDS_THRESHOLD: The source DB size in Bytes exceeds a
        certain threshold. The migration might require an increase of quota,
        or might not be supported.
      EXISTING_CONFLICTING_DATABASES: The destination DB contains existing
        databases that are conflicting with those in the source DB.
      PARALLEL_IMPORT_INSUFFICIENT_PRIVILEGE: Insufficient privilege to enable
        the parallelism configuration.
      EXISTING_DATA: The destination instance contains existing data or user
        defined entities (for example databases, tables, or functions). You
        can only migrate to empty instances. Clear your destination instance
        and retry the migration job.
      SOURCE_MAX_SUBSCRIPTIONS: The migration job is configured to use max
        number of subscriptions to migrate data from the source to the
        destination.
    """
        ERROR_CODE_UNSPECIFIED = 0
        CONNECTION_FAILURE = 1
        AUTHENTICATION_FAILURE = 2
        INVALID_CONNECTION_PROFILE_CONFIG = 3
        VERSION_INCOMPATIBILITY = 4
        CONNECTION_PROFILE_TYPES_INCOMPATIBILITY = 5
        UNSUPPORTED_GTID_MODE = 6
        UNSUPPORTED_DEFINER = 7
        CANT_RESTART_RUNNING_MIGRATION = 8
        TABLES_WITH_LIMITED_SUPPORT = 9
        UNSUPPORTED_DATABASE_LOCALE = 10
        UNSUPPORTED_DATABASE_FDW_CONFIG = 11
        ERROR_RDBMS = 12
        SOURCE_SIZE_EXCEEDS_THRESHOLD = 13
        EXISTING_CONFLICTING_DATABASES = 14
        PARALLEL_IMPORT_INSUFFICIENT_PRIVILEGE = 15
        EXISTING_DATA = 16
        SOURCE_MAX_SUBSCRIPTIONS = 17
    errorCode = _messages.EnumField('ErrorCodeValueValuesEnum', 1)
    errorDetailMessage = _messages.StringField(2)
    errorMessage = _messages.StringField(3)