from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SqlExternalSyncSettingError(_messages.Message):
    """External primary instance migration setting error/warning.

  Enums:
    TypeValueValuesEnum: Identifies the specific error that occurred.

  Fields:
    detail: Additional information about the error encountered.
    kind: Can be `sql#externalSyncSettingError` or
      `sql#externalSyncSettingWarning`.
    type: Identifies the specific error that occurred.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """Identifies the specific error that occurred.

    Values:
      SQL_EXTERNAL_SYNC_SETTING_ERROR_TYPE_UNSPECIFIED: <no description>
      CONNECTION_FAILURE: <no description>
      BINLOG_NOT_ENABLED: <no description>
      INCOMPATIBLE_DATABASE_VERSION: <no description>
      REPLICA_ALREADY_SETUP: <no description>
      INSUFFICIENT_PRIVILEGE: The replication user is missing privileges that
        are required.
      UNSUPPORTED_MIGRATION_TYPE: Unsupported migration type.
      NO_PGLOGICAL_INSTALLED: No pglogical extension installed on databases,
        applicable for postgres.
      PGLOGICAL_NODE_ALREADY_EXISTS: pglogical node already exists on
        databases, applicable for postgres.
      INVALID_WAL_LEVEL: The value of parameter wal_level is not set to
        logical.
      INVALID_SHARED_PRELOAD_LIBRARY: The value of parameter
        shared_preload_libraries does not include pglogical.
      INSUFFICIENT_MAX_REPLICATION_SLOTS: The value of parameter
        max_replication_slots is not sufficient.
      INSUFFICIENT_MAX_WAL_SENDERS: The value of parameter max_wal_senders is
        not sufficient.
      INSUFFICIENT_MAX_WORKER_PROCESSES: The value of parameter
        max_worker_processes is not sufficient.
      UNSUPPORTED_EXTENSIONS: Extensions installed are either not supported or
        having unsupported versions
      INVALID_RDS_LOGICAL_REPLICATION: The value of parameter
        rds.logical_replication is not set to 1.
      INVALID_LOGGING_SETUP: The primary instance logging setup doesn't allow
        EM sync.
      INVALID_DB_PARAM: The primary instance database parameter setup doesn't
        allow EM sync.
      UNSUPPORTED_GTID_MODE: The gtid_mode is not supported, applicable for
        MySQL.
      SQLSERVER_AGENT_NOT_RUNNING: SQL Server Agent is not running.
      UNSUPPORTED_TABLE_DEFINITION: The table definition is not support due to
        missing primary key or replica identity, applicable for postgres.
      UNSUPPORTED_DEFINER: The customer has a definer that will break EM
        setup.
      SQLSERVER_SERVERNAME_MISMATCH: SQL Server @@SERVERNAME does not match
        actual host name.
      PRIMARY_ALREADY_SETUP: The primary instance has been setup and will fail
        the setup.
      UNSUPPORTED_BINLOG_FORMAT: The primary instance has unsupported binary
        log format.
      BINLOG_RETENTION_SETTING: The primary instance's binary log retention
        setting.
      UNSUPPORTED_STORAGE_ENGINE: The primary instance has tables with
        unsupported storage engine.
      LIMITED_SUPPORT_TABLES: Source has tables with limited support eg:
        PostgreSQL tables without primary keys.
      EXISTING_DATA_IN_REPLICA: The replica instance contains existing data.
      MISSING_OPTIONAL_PRIVILEGES: The replication user is missing privileges
        that are optional.
      RISKY_BACKUP_ADMIN_PRIVILEGE: Additional BACKUP_ADMIN privilege is
        granted to the replication user which may lock source MySQL 8 instance
        for DDLs during initial sync.
      INSUFFICIENT_GCS_PERMISSIONS: The Cloud Storage bucket is missing
        necessary permissions.
      INVALID_FILE_INFO: The Cloud Storage bucket has an error in the file or
        contains invalid file information.
      UNSUPPORTED_DATABASE_SETTINGS: The source instance has unsupported
        database settings for migration.
      MYSQL_PARALLEL_IMPORT_INSUFFICIENT_PRIVILEGE: The replication user is
        missing parallel import specific privileges. (e.g. LOCK TABLES) for
        MySQL.
      LOCAL_INFILE_OFF: The global variable local_infile is off on external
        server replica.
      TURN_ON_PITR_AFTER_PROMOTE: This code instructs customers to turn on
        point-in-time recovery manually for the instance after promoting the
        Cloud SQL for PostgreSQL instance.
      INCOMPATIBLE_DATABASE_MINOR_VERSION: The minor version of replica
        database is incompatible with the source.
      SOURCE_MAX_SUBSCRIPTIONS: This warning message indicates that Cloud SQL
        uses the maximum number of subscriptions to migrate data from the
        source to the destination.
      UNABLE_TO_VERIFY_DEFINERS: Unable to verify definers on the source for
        MySQL.
      SUBSCRIPTION_CALCULATION_STATUS: If a time out occurs while the
        subscription counts are calculated, then this value is set to 1.
        Otherwise, this value is set to 2.
      PG_SUBSCRIPTION_COUNT: Count of subscriptions needed to sync source data
        for PostgreSQL database.
      PG_SYNC_PARALLEL_LEVEL: Final parallel level that is used to do
        migration.
      INSUFFICIENT_DISK_SIZE: The disk size of the replica instance is smaller
        than the data size of the source instance.
      INSUFFICIENT_MACHINE_TIER: The data size of the source instance is
        greater than 1 TB, the number of cores of the replica instance is less
        than 8, and the memory of the replica is less than 32 GB.
    """
        SQL_EXTERNAL_SYNC_SETTING_ERROR_TYPE_UNSPECIFIED = 0
        CONNECTION_FAILURE = 1
        BINLOG_NOT_ENABLED = 2
        INCOMPATIBLE_DATABASE_VERSION = 3
        REPLICA_ALREADY_SETUP = 4
        INSUFFICIENT_PRIVILEGE = 5
        UNSUPPORTED_MIGRATION_TYPE = 6
        NO_PGLOGICAL_INSTALLED = 7
        PGLOGICAL_NODE_ALREADY_EXISTS = 8
        INVALID_WAL_LEVEL = 9
        INVALID_SHARED_PRELOAD_LIBRARY = 10
        INSUFFICIENT_MAX_REPLICATION_SLOTS = 11
        INSUFFICIENT_MAX_WAL_SENDERS = 12
        INSUFFICIENT_MAX_WORKER_PROCESSES = 13
        UNSUPPORTED_EXTENSIONS = 14
        INVALID_RDS_LOGICAL_REPLICATION = 15
        INVALID_LOGGING_SETUP = 16
        INVALID_DB_PARAM = 17
        UNSUPPORTED_GTID_MODE = 18
        SQLSERVER_AGENT_NOT_RUNNING = 19
        UNSUPPORTED_TABLE_DEFINITION = 20
        UNSUPPORTED_DEFINER = 21
        SQLSERVER_SERVERNAME_MISMATCH = 22
        PRIMARY_ALREADY_SETUP = 23
        UNSUPPORTED_BINLOG_FORMAT = 24
        BINLOG_RETENTION_SETTING = 25
        UNSUPPORTED_STORAGE_ENGINE = 26
        LIMITED_SUPPORT_TABLES = 27
        EXISTING_DATA_IN_REPLICA = 28
        MISSING_OPTIONAL_PRIVILEGES = 29
        RISKY_BACKUP_ADMIN_PRIVILEGE = 30
        INSUFFICIENT_GCS_PERMISSIONS = 31
        INVALID_FILE_INFO = 32
        UNSUPPORTED_DATABASE_SETTINGS = 33
        MYSQL_PARALLEL_IMPORT_INSUFFICIENT_PRIVILEGE = 34
        LOCAL_INFILE_OFF = 35
        TURN_ON_PITR_AFTER_PROMOTE = 36
        INCOMPATIBLE_DATABASE_MINOR_VERSION = 37
        SOURCE_MAX_SUBSCRIPTIONS = 38
        UNABLE_TO_VERIFY_DEFINERS = 39
        SUBSCRIPTION_CALCULATION_STATUS = 40
        PG_SUBSCRIPTION_COUNT = 41
        PG_SYNC_PARALLEL_LEVEL = 42
        INSUFFICIENT_DISK_SIZE = 43
        INSUFFICIENT_MACHINE_TIER = 44
    detail = _messages.StringField(1)
    kind = _messages.StringField(2)
    type = _messages.EnumField('TypeValueValuesEnum', 3)