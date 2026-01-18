from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleFirestoreAdminV1Database(_messages.Message):
    """A Cloud Firestore Database.

  Enums:
    AppEngineIntegrationModeValueValuesEnum: The App Engine integration mode
      to use for this database.
    ConcurrencyModeValueValuesEnum: The concurrency control mode to use for
      this database.
    DeleteProtectionStateValueValuesEnum: State of delete protection for the
      database.
    PointInTimeRecoveryEnablementValueValuesEnum: Whether to enable the PITR
      feature on this database.
    TypeValueValuesEnum: The type of the database. See
      https://cloud.google.com/datastore/docs/firestore-or-datastore for
      information about how to choose.

  Fields:
    appEngineIntegrationMode: The App Engine integration mode to use for this
      database.
    cmekConfig: Optional. Presence indicates CMEK is enabled for this
      database.
    concurrencyMode: The concurrency control mode to use for this database.
    createTime: Output only. The timestamp at which this database was created.
      Databases created before 2016 do not populate create_time.
    deleteProtectionState: State of delete protection for the database.
    earliestVersionTime: Output only. The earliest timestamp at which older
      versions of the data can be read from the database. See
      [version_retention_period] above; this field is populated with `now -
      version_retention_period`. This value is continuously updated, and
      becomes stale the moment it is queried. If you are using this value to
      recover data, make sure to account for the time from the moment when the
      value is queried to the moment when you initiate the recovery.
    etag: This checksum is computed by the server based on the value of other
      fields, and may be sent on update and delete requests to ensure the
      client has an up-to-date value before proceeding.
    keyPrefix: Output only. The key_prefix for this database. This key_prefix
      is used, in combination with the project id ("~") to construct the
      application id that is returned from the Cloud Datastore APIs in Google
      App Engine first generation runtimes. This value may be empty in which
      case the appid to use for URL-encoded keys is the project_id (eg: foo
      instead of v~foo).
    locationId: The location of the database. Available locations are listed
      at https://cloud.google.com/firestore/docs/locations.
    name: The resource name of the Database. Format:
      `projects/{project}/databases/{database}`
    pointInTimeRecoveryEnablement: Whether to enable the PITR feature on this
      database.
    type: The type of the database. See
      https://cloud.google.com/datastore/docs/firestore-or-datastore for
      information about how to choose.
    uid: Output only. The system-generated UUID4 for this Database.
    updateTime: Output only. The timestamp at which this database was most
      recently updated. Note this only includes updates to the database
      resource and not data contained by the database.
    versionRetentionPeriod: Output only. The period during which past versions
      of data are retained in the database. Any read or query can specify a
      `read_time` within this window, and will read the state of the database
      at that time. If the PITR feature is enabled, the retention period is 7
      days. Otherwise, the retention period is 1 hour.
  """

    class AppEngineIntegrationModeValueValuesEnum(_messages.Enum):
        """The App Engine integration mode to use for this database.

    Values:
      APP_ENGINE_INTEGRATION_MODE_UNSPECIFIED: Not used.
      ENABLED: If an App Engine application exists in the same region as this
        database, App Engine configuration will impact this database. This
        includes disabling of the application & database, as well as disabling
        writes to the database.
      DISABLED: App Engine has no effect on the ability of this database to
        serve requests. This is the default setting for databases created with
        the Firestore API.
    """
        APP_ENGINE_INTEGRATION_MODE_UNSPECIFIED = 0
        ENABLED = 1
        DISABLED = 2

    class ConcurrencyModeValueValuesEnum(_messages.Enum):
        """The concurrency control mode to use for this database.

    Values:
      CONCURRENCY_MODE_UNSPECIFIED: Not used.
      OPTIMISTIC: Use optimistic concurrency control by default. This mode is
        available for Cloud Firestore databases.
      PESSIMISTIC: Use pessimistic concurrency control by default. This mode
        is available for Cloud Firestore databases. This is the default
        setting for Cloud Firestore.
      OPTIMISTIC_WITH_ENTITY_GROUPS: Use optimistic concurrency control with
        entity groups by default. This is the only available mode for Cloud
        Datastore. This mode is also available for Cloud Firestore with
        Datastore Mode but is not recommended.
    """
        CONCURRENCY_MODE_UNSPECIFIED = 0
        OPTIMISTIC = 1
        PESSIMISTIC = 2
        OPTIMISTIC_WITH_ENTITY_GROUPS = 3

    class DeleteProtectionStateValueValuesEnum(_messages.Enum):
        """State of delete protection for the database.

    Values:
      DELETE_PROTECTION_STATE_UNSPECIFIED: The default value. Delete
        protection type is not specified
      DELETE_PROTECTION_DISABLED: Delete protection is disabled
      DELETE_PROTECTION_ENABLED: Delete protection is enabled
    """
        DELETE_PROTECTION_STATE_UNSPECIFIED = 0
        DELETE_PROTECTION_DISABLED = 1
        DELETE_PROTECTION_ENABLED = 2

    class PointInTimeRecoveryEnablementValueValuesEnum(_messages.Enum):
        """Whether to enable the PITR feature on this database.

    Values:
      POINT_IN_TIME_RECOVERY_ENABLEMENT_UNSPECIFIED: Not used.
      POINT_IN_TIME_RECOVERY_ENABLED: Reads are supported on selected versions
        of the data from within the past 7 days: * Reads against any timestamp
        within the past hour * Reads against 1-minute snapshots beyond 1 hour
        and within 7 days `version_retention_period` and
        `earliest_version_time` can be used to determine the supported
        versions.
      POINT_IN_TIME_RECOVERY_DISABLED: Reads are supported on any version of
        the data from within the past 1 hour.
    """
        POINT_IN_TIME_RECOVERY_ENABLEMENT_UNSPECIFIED = 0
        POINT_IN_TIME_RECOVERY_ENABLED = 1
        POINT_IN_TIME_RECOVERY_DISABLED = 2

    class TypeValueValuesEnum(_messages.Enum):
        """The type of the database. See
    https://cloud.google.com/datastore/docs/firestore-or-datastore for
    information about how to choose.

    Values:
      DATABASE_TYPE_UNSPECIFIED: The default value. This value is used if the
        database type is omitted.
      FIRESTORE_NATIVE: Firestore Native Mode
      DATASTORE_MODE: Firestore in Datastore Mode.
    """
        DATABASE_TYPE_UNSPECIFIED = 0
        FIRESTORE_NATIVE = 1
        DATASTORE_MODE = 2
    appEngineIntegrationMode = _messages.EnumField('AppEngineIntegrationModeValueValuesEnum', 1)
    cmekConfig = _messages.MessageField('GoogleFirestoreAdminV1CmekConfig', 2)
    concurrencyMode = _messages.EnumField('ConcurrencyModeValueValuesEnum', 3)
    createTime = _messages.StringField(4)
    deleteProtectionState = _messages.EnumField('DeleteProtectionStateValueValuesEnum', 5)
    earliestVersionTime = _messages.StringField(6)
    etag = _messages.StringField(7)
    keyPrefix = _messages.StringField(8)
    locationId = _messages.StringField(9)
    name = _messages.StringField(10)
    pointInTimeRecoveryEnablement = _messages.EnumField('PointInTimeRecoveryEnablementValueValuesEnum', 11)
    type = _messages.EnumField('TypeValueValuesEnum', 12)
    uid = _messages.StringField(13)
    updateTime = _messages.StringField(14)
    versionRetentionPeriod = _messages.StringField(15)