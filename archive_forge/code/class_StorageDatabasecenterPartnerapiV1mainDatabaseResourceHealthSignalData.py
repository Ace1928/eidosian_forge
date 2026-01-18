from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StorageDatabasecenterPartnerapiV1mainDatabaseResourceHealthSignalData(_messages.Message):
    """Common model for database resource health signal data.

  Enums:
    ProviderValueValuesEnum: Cloud provider name. Ex:
      GCP/AWS/Azure/OnPrem/SelfManaged
    SignalClassValueValuesEnum: Required. The class of the signal, such as if
      it's a THREAT or VULNERABILITY.
    SignalTypeValueValuesEnum: Required. Type of signal, for example,
      `AVAILABLE_IN_MULTIPLE_ZONES`, `LOGGING_MOST_ERRORS`, etc.
    StateValueValuesEnum:

  Messages:
    AdditionalMetadataValue: Any other additional metadata

  Fields:
    additionalMetadata: Any other additional metadata
    compliance: Industry standards associated with this signal; if this signal
      is an issue, that could be a violation of the associated industry
      standard(s). For example, AUTO_BACKUP_DISABLED signal is associated with
      CIS GCP 1.1, CIS GCP 1.2, CIS GCP 1.3, NIST 800-53 and ISO-27001
      compliance standards. If a database resource does not have automated
      backup enable, it will violate these following industry standards.
    description: Description associated with signal
    eventTime: Required. The last time at which the event described by this
      signal took place
    externalUri: The external-uri of the signal, using which more information
      about this signal can be obtained. In GCP, this will take user to SCC
      page to get more details about signals.
    name: Required. The name of the signal, ex: PUBLIC_SQL_INSTANCE,
      SQL_LOG_ERROR_VERBOSITY etc.
    provider: Cloud provider name. Ex: GCP/AWS/Azure/OnPrem/SelfManaged
    resourceContainer: Closest parent container of this resource. In GCP,
      'container' refers to a Cloud Resource Manager project. It must be
      resource name of a Cloud Resource Manager project with the format of
      "provider//", such as "projects/123". For GCP provided resources, number
      should be project number.
    resourceName: Required. Database resource name associated with the signal.
      Resource name to follow CAIS resource_name format as noted here
      go/condor-common-datamodel
    signalClass: Required. The class of the signal, such as if it's a THREAT
      or VULNERABILITY.
    signalId: Required. Unique identifier for the signal. This is an unique id
      which would be mainatined by partner to identify a signal.
    signalType: Required. Type of signal, for example,
      `AVAILABLE_IN_MULTIPLE_ZONES`, `LOGGING_MOST_ERRORS`, etc.
    state: A StateValueValuesEnum attribute.
  """

    class ProviderValueValuesEnum(_messages.Enum):
        """Cloud provider name. Ex: GCP/AWS/Azure/OnPrem/SelfManaged

    Values:
      PROVIDER_UNSPECIFIED: <no description>
      GCP: Google cloud platform provider
      AWS: Amazon web service
      AZURE: Azure web service
      ONPREM: On-prem database resources.
      SELFMANAGED: Self-managed database provider. These are resources on a
        cloud platform, e.g., database resource installed in a GCE VM, but not
        a managed database service.
      PROVIDER_OTHER: For the rest of the other categories. Other refers to
        the rest of other database service providers, this could be smaller
        cloud provider. This needs to be provided when the provider is known,
        but it is not present in the existing set of enum values.
    """
        PROVIDER_UNSPECIFIED = 0
        GCP = 1
        AWS = 2
        AZURE = 3
        ONPREM = 4
        SELFMANAGED = 5
        PROVIDER_OTHER = 6

    class SignalClassValueValuesEnum(_messages.Enum):
        """Required. The class of the signal, such as if it's a THREAT or
    VULNERABILITY.

    Values:
      CLASS_UNSPECIFIED: Unspecified signal class.
      THREAT: Describes unwanted or malicious activity.
      VULNERABILITY: Describes a potential weakness in software that increases
        risk to Confidentiality & Integrity & Availability.
      MISCONFIGURATION: Describes a potential weakness in cloud resource/asset
        configuration that increases risk.
      OBSERVATION: Describes a security observation that is for informational
        purposes.
      ERROR: Describes an error that prevents some SCC functionality.
    """
        CLASS_UNSPECIFIED = 0
        THREAT = 1
        VULNERABILITY = 2
        MISCONFIGURATION = 3
        OBSERVATION = 4
        ERROR = 5

    class SignalTypeValueValuesEnum(_messages.Enum):
        """Required. Type of signal, for example, `AVAILABLE_IN_MULTIPLE_ZONES`,
    `LOGGING_MOST_ERRORS`, etc.

    Values:
      SIGNAL_TYPE_UNSPECIFIED: Unspecified.
      SIGNAL_TYPE_NOT_PROTECTED_BY_AUTOMATIC_FAILOVER: Represents if a
        resource is protected by automatic failover. Checks for resources that
        are configured to have redundancy within a region that enables
        automatic failover.
      SIGNAL_TYPE_GROUP_NOT_REPLICATING_ACROSS_REGIONS: Represents if a group
        is replicating across regions. Checks for resources that are
        configured to have redundancy, and ongoing replication, across
        regions.
      SIGNAL_TYPE_NOT_AVAILABLE_IN_MULTIPLE_ZONES: Represents if the resource
        is available in multiple zones or not.
      SIGNAL_TYPE_NOT_AVAILABLE_IN_MULTIPLE_REGIONS: Represents if a resource
        is available in multiple regions.
      SIGNAL_TYPE_NO_PROMOTABLE_REPLICA: Represents if a resource has a
        promotable replica.
      SIGNAL_TYPE_NO_AUTOMATED_BACKUP_POLICY: Represents if a resource has an
        automated backup policy.
      SIGNAL_TYPE_SHORT_BACKUP_RETENTION: Represents if a resources has a
        short backup retention period.
      SIGNAL_TYPE_LAST_BACKUP_FAILED: Represents if the last backup of a
        resource failed.
      SIGNAL_TYPE_LAST_BACKUP_OLD: Represents if the last backup of a resource
        is older than some threshold value.
      SIGNAL_TYPE_VIOLATES_CIS_GCP_FOUNDATION_2_0: Represents if a resource
        violates CIS GCP Foundation 2.0.
      SIGNAL_TYPE_VIOLATES_CIS_GCP_FOUNDATION_1_3: Represents if a resource
        violates CIS GCP Foundation 1.3.
      SIGNAL_TYPE_VIOLATES_CIS_GCP_FOUNDATION_1_2: Represents if a resource
        violates CIS GCP Foundation 1.2.
      SIGNAL_TYPE_VIOLATES_CIS_GCP_FOUNDATION_1_1: Represents if a resource
        violates CIS GCP Foundation 1.1.
      SIGNAL_TYPE_VIOLATES_CIS_GCP_FOUNDATION_1_0: Represents if a resource
        violates CIS GCP Foundation 1.0.
      SIGNAL_TYPE_VIOLATES_NIST_800_53: Represents if a resource violates NIST
        800-53.
      SIGNAL_TYPE_VIOLATES_ISO_27001: Represents if a resource violates
        ISO-27001.
      SIGNAL_TYPE_VIOLATES_PCI_DSS_V3_2_1: Represents if a resource violates
        PCI-DSS v3.2.1.
      SIGNAL_TYPE_LOGS_NOT_OPTIMIZED_FOR_TROUBLESHOOTING: Represents if
        log_checkpoints database flag for a Cloud SQL for PostgreSQL instance
        is not set to on.
      SIGNAL_TYPE_QUERY_DURATIONS_NOT_LOGGED: Represents if the log_duration
        database flag for a Cloud SQL for PostgreSQL instance is not set to
        on.
      SIGNAL_TYPE_VERBOSE_ERROR_LOGGING: Represents if the log_error_verbosity
        database flag for a Cloud SQL for PostgreSQL instance is not set to
        default or stricter (default or terse).
      SIGNAL_TYPE_QUERY_LOCK_WAITS_NOT_LOGGED: Represents if the
        log_lock_waits database flag for a Cloud SQL for PostgreSQL instance
        is not set to on.
      SIGNAL_TYPE_LOGGING_MOST_ERRORS: Represents if the
        log_min_error_statement database flag for a Cloud SQL for PostgreSQL
        instance is not set appropriately.
      SIGNAL_TYPE_LOGGING_ONLY_CRITICAL_ERRORS: Represents if the
        log_min_error_statement database flag for a Cloud SQL for PostgreSQL
        instance does not have an appropriate severity level.
      SIGNAL_TYPE_MINIMAL_ERROR_LOGGING: Represents if the log_min_messages
        database flag for a Cloud SQL for PostgreSQL instance is not set to
        warning or another recommended value.
      SIGNAL_TYPE_QUERY_STATISTICS_LOGGED: Represents if the databaseFlags
        property of instance metadata for the log_executor_status field is set
        to on.
      SIGNAL_TYPE_EXCESSIVE_LOGGING_OF_CLIENT_HOSTNAME: Represents if the
        log_hostname database flag for a Cloud SQL for PostgreSQL instance is
        not set to off.
      SIGNAL_TYPE_EXCESSIVE_LOGGING_OF_PARSER_STATISTICS: Represents if the
        log_parser_stats database flag for a Cloud SQL for PostgreSQL instance
        is not set to off.
      SIGNAL_TYPE_EXCESSIVE_LOGGING_OF_PLANNER_STATISTICS: Represents if the
        log_planner_stats database flag for a Cloud SQL for PostgreSQL
        instance is not set to off.
      SIGNAL_TYPE_NOT_LOGGING_ONLY_DDL_STATEMENTS: Represents if the
        log_statement database flag for a Cloud SQL for PostgreSQL instance is
        not set to DDL (all data definition statements).
      SIGNAL_TYPE_LOGGING_QUERY_STATISTICS: Represents if the
        log_statement_stats database flag for a Cloud SQL for PostgreSQL
        instance is not set to off.
      SIGNAL_TYPE_NOT_LOGGING_TEMPORARY_FILES: Represents if the
        log_temp_files database flag for a Cloud SQL for PostgreSQL instance
        is not set to "0". (NOTE: 0 = ON)
      SIGNAL_TYPE_CONNECTION_MAX_NOT_CONFIGURED: Represents if the user
        connections database flag for a Cloud SQL for SQL Server instance is
        configured.
      SIGNAL_TYPE_USER_OPTIONS_CONFIGURED: Represents if the user options
        database flag for Cloud SQL SQL Server instance is configured or not.
      SIGNAL_TYPE_EXPOSED_TO_PUBLIC_ACCESS: Represents if a resource is
        exposed to public access.
      SIGNAL_TYPE_UNENCRYPTED_CONNECTIONS: Represents if a resources requires
        all incoming connections to use SSL or not.
      SIGNAL_TYPE_NO_ROOT_PASSWORD: Represents if a Cloud SQL database has a
        password configured for the root account or not.
      SIGNAL_TYPE_WEAK_ROOT_PASSWORD: Represents if a Cloud SQL database has a
        weak password configured for the root account.
      SIGNAL_TYPE_ENCRYPTION_KEY_NOT_CUSTOMER_MANAGED: Represents if a SQL
        database instance is not encrypted with customer-managed encryption
        keys (CMEK).
      SIGNAL_TYPE_SERVER_AUTHENTICATION_NOT_REQUIRED: Represents if The
        contained database authentication database flag for a Cloud SQL for
        SQL Server instance is not set to off.
      SIGNAL_TYPE_EXPOSED_BY_OWNERSHIP_CHAINING: Represents if the
        cross_db_ownership_chaining database flag for a Cloud SQL for SQL
        Server instance is not set to off.
      SIGNAL_TYPE_EXPOSED_TO_EXTERNAL_SCRIPTS: Represents if he external
        scripts enabled database flag for a Cloud SQL for SQL Server instance
        is not set to off.
      SIGNAL_TYPE_EXPOSED_TO_LOCAL_DATA_LOADS: Represents if the local_infile
        database flag for a Cloud SQL for MySQL instance is not set to off.
      SIGNAL_TYPE_CONNECTION_ATTEMPTS_NOT_LOGGED: Represents if the
        log_connections database flag for a Cloud SQL for PostgreSQL instance
        is not set to on.
      SIGNAL_TYPE_DISCONNECTIONS_NOT_LOGGED: Represents if the
        log_disconnections database flag for a Cloud SQL for PostgreSQL
        instance is not set to on.
      SIGNAL_TYPE_LOGGING_EXCESSIVE_STATEMENT_INFO: Represents if the
        log_min_duration_statement database flag for a Cloud SQL for
        PostgreSQL instance is not set to -1.
      SIGNAL_TYPE_EXPOSED_TO_REMOTE_ACCESS: Represents if the remote access
        database flag for a Cloud SQL for SQL Server instance is not set to
        off.
      SIGNAL_TYPE_DATABASE_NAMES_EXPOSED: Represents if the skip_show_database
        database flag for a Cloud SQL for MySQL instance is not set to on.
      SIGNAL_TYPE_SENSITIVE_TRACE_INFO_NOT_MASKED: Represents if the 3625
        (trace flag) database flag for a Cloud SQL for SQL Server instance is
        not set to on.
      SIGNAL_TYPE_PUBLIC_IP_ENABLED: Represents if public IP is enabled.
      SIGNAL_TYPE_IDLE: Represents Idle instance helps to reduce costs.
      SIGNAL_TYPE_OVERPROVISIONED: Represents instances that are unnecessarily
        large for given workload.
      SIGNAL_TYPE_HIGH_NUMBER_OF_OPEN_TABLES: Represents high number of
        concurrently opened tables.
      SIGNAL_TYPE_HIGH_NUMBER_OF_TABLES: Represents high table count close to
        SLA limit.
      SIGNAL_TYPE_HIGH_TRANSACTION_ID_UTILIZATION: Represents high number of
        unvacuumed transactions
      SIGNAL_TYPE_UNDERPROVISIONED: Represents need for more CPU and/or memory
      SIGNAL_TYPE_OUT_OF_DISK: Represents out of disk.
      SIGNAL_TYPE_SERVER_CERTIFICATE_NEAR_EXPIRY: Represents server
        certificate is near expiry.
      SIGNAL_TYPE_DATABASE_AUDITING_DISABLED: Represents database auditing is
        disabled.
      SIGNAL_TYPE_RESTRICT_AUTHORIZED_NETWORKS: Represents not restricted to
        authorized networks.
      SIGNAL_TYPE_VIOLATE_POLICY_RESTRICT_PUBLIC_IP: Represents violate org
        policy restrict public ip.
      SIGNAL_TYPE_QUOTA_LIMIT: Cluster nearing quota limit
      SIGNAL_TYPE_NO_PASSWORD_POLICY: No password policy set on resources
      SIGNAL_TYPE_CONNECTIONS_PERFORMANCE_IMPACT: Performance impact of
        connections settings
      SIGNAL_TYPE_TMP_TABLES_PERFORMANCE_IMPACT: Performance impact of
        temporary tables settings
      SIGNAL_TYPE_TRANS_LOGS_PERFORMANCE_IMPACT: Performance impact of
        transaction logs settings
      SIGNAL_TYPE_HIGH_JOINS_WITHOUT_INDEXES: Performance impact of high joins
        without indexes
    """
        SIGNAL_TYPE_UNSPECIFIED = 0
        SIGNAL_TYPE_NOT_PROTECTED_BY_AUTOMATIC_FAILOVER = 1
        SIGNAL_TYPE_GROUP_NOT_REPLICATING_ACROSS_REGIONS = 2
        SIGNAL_TYPE_NOT_AVAILABLE_IN_MULTIPLE_ZONES = 3
        SIGNAL_TYPE_NOT_AVAILABLE_IN_MULTIPLE_REGIONS = 4
        SIGNAL_TYPE_NO_PROMOTABLE_REPLICA = 5
        SIGNAL_TYPE_NO_AUTOMATED_BACKUP_POLICY = 6
        SIGNAL_TYPE_SHORT_BACKUP_RETENTION = 7
        SIGNAL_TYPE_LAST_BACKUP_FAILED = 8
        SIGNAL_TYPE_LAST_BACKUP_OLD = 9
        SIGNAL_TYPE_VIOLATES_CIS_GCP_FOUNDATION_2_0 = 10
        SIGNAL_TYPE_VIOLATES_CIS_GCP_FOUNDATION_1_3 = 11
        SIGNAL_TYPE_VIOLATES_CIS_GCP_FOUNDATION_1_2 = 12
        SIGNAL_TYPE_VIOLATES_CIS_GCP_FOUNDATION_1_1 = 13
        SIGNAL_TYPE_VIOLATES_CIS_GCP_FOUNDATION_1_0 = 14
        SIGNAL_TYPE_VIOLATES_NIST_800_53 = 15
        SIGNAL_TYPE_VIOLATES_ISO_27001 = 16
        SIGNAL_TYPE_VIOLATES_PCI_DSS_V3_2_1 = 17
        SIGNAL_TYPE_LOGS_NOT_OPTIMIZED_FOR_TROUBLESHOOTING = 18
        SIGNAL_TYPE_QUERY_DURATIONS_NOT_LOGGED = 19
        SIGNAL_TYPE_VERBOSE_ERROR_LOGGING = 20
        SIGNAL_TYPE_QUERY_LOCK_WAITS_NOT_LOGGED = 21
        SIGNAL_TYPE_LOGGING_MOST_ERRORS = 22
        SIGNAL_TYPE_LOGGING_ONLY_CRITICAL_ERRORS = 23
        SIGNAL_TYPE_MINIMAL_ERROR_LOGGING = 24
        SIGNAL_TYPE_QUERY_STATISTICS_LOGGED = 25
        SIGNAL_TYPE_EXCESSIVE_LOGGING_OF_CLIENT_HOSTNAME = 26
        SIGNAL_TYPE_EXCESSIVE_LOGGING_OF_PARSER_STATISTICS = 27
        SIGNAL_TYPE_EXCESSIVE_LOGGING_OF_PLANNER_STATISTICS = 28
        SIGNAL_TYPE_NOT_LOGGING_ONLY_DDL_STATEMENTS = 29
        SIGNAL_TYPE_LOGGING_QUERY_STATISTICS = 30
        SIGNAL_TYPE_NOT_LOGGING_TEMPORARY_FILES = 31
        SIGNAL_TYPE_CONNECTION_MAX_NOT_CONFIGURED = 32
        SIGNAL_TYPE_USER_OPTIONS_CONFIGURED = 33
        SIGNAL_TYPE_EXPOSED_TO_PUBLIC_ACCESS = 34
        SIGNAL_TYPE_UNENCRYPTED_CONNECTIONS = 35
        SIGNAL_TYPE_NO_ROOT_PASSWORD = 36
        SIGNAL_TYPE_WEAK_ROOT_PASSWORD = 37
        SIGNAL_TYPE_ENCRYPTION_KEY_NOT_CUSTOMER_MANAGED = 38
        SIGNAL_TYPE_SERVER_AUTHENTICATION_NOT_REQUIRED = 39
        SIGNAL_TYPE_EXPOSED_BY_OWNERSHIP_CHAINING = 40
        SIGNAL_TYPE_EXPOSED_TO_EXTERNAL_SCRIPTS = 41
        SIGNAL_TYPE_EXPOSED_TO_LOCAL_DATA_LOADS = 42
        SIGNAL_TYPE_CONNECTION_ATTEMPTS_NOT_LOGGED = 43
        SIGNAL_TYPE_DISCONNECTIONS_NOT_LOGGED = 44
        SIGNAL_TYPE_LOGGING_EXCESSIVE_STATEMENT_INFO = 45
        SIGNAL_TYPE_EXPOSED_TO_REMOTE_ACCESS = 46
        SIGNAL_TYPE_DATABASE_NAMES_EXPOSED = 47
        SIGNAL_TYPE_SENSITIVE_TRACE_INFO_NOT_MASKED = 48
        SIGNAL_TYPE_PUBLIC_IP_ENABLED = 49
        SIGNAL_TYPE_IDLE = 50
        SIGNAL_TYPE_OVERPROVISIONED = 51
        SIGNAL_TYPE_HIGH_NUMBER_OF_OPEN_TABLES = 52
        SIGNAL_TYPE_HIGH_NUMBER_OF_TABLES = 53
        SIGNAL_TYPE_HIGH_TRANSACTION_ID_UTILIZATION = 54
        SIGNAL_TYPE_UNDERPROVISIONED = 55
        SIGNAL_TYPE_OUT_OF_DISK = 56
        SIGNAL_TYPE_SERVER_CERTIFICATE_NEAR_EXPIRY = 57
        SIGNAL_TYPE_DATABASE_AUDITING_DISABLED = 58
        SIGNAL_TYPE_RESTRICT_AUTHORIZED_NETWORKS = 59
        SIGNAL_TYPE_VIOLATE_POLICY_RESTRICT_PUBLIC_IP = 60
        SIGNAL_TYPE_QUOTA_LIMIT = 61
        SIGNAL_TYPE_NO_PASSWORD_POLICY = 62
        SIGNAL_TYPE_CONNECTIONS_PERFORMANCE_IMPACT = 63
        SIGNAL_TYPE_TMP_TABLES_PERFORMANCE_IMPACT = 64
        SIGNAL_TYPE_TRANS_LOGS_PERFORMANCE_IMPACT = 65
        SIGNAL_TYPE_HIGH_JOINS_WITHOUT_INDEXES = 66

    class StateValueValuesEnum(_messages.Enum):
        """StateValueValuesEnum enum type.

    Values:
      STATE_UNSPECIFIED: Unspecified state.
      ACTIVE: The signal requires attention and has not been addressed yet.
      RESOLVED: The signal has been fixed, triaged as a non-issue or otherwise
        addressed and is no longer active.
      MUTED: The signal has been muted.
    """
        STATE_UNSPECIFIED = 0
        ACTIVE = 1
        RESOLVED = 2
        MUTED = 3

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AdditionalMetadataValue(_messages.Message):
        """Any other additional metadata

    Messages:
      AdditionalProperty: An additional property for a AdditionalMetadataValue
        object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a AdditionalMetadataValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    additionalMetadata = _messages.MessageField('AdditionalMetadataValue', 1)
    compliance = _messages.MessageField('StorageDatabasecenterPartnerapiV1mainCompliance', 2, repeated=True)
    description = _messages.StringField(3)
    eventTime = _messages.StringField(4)
    externalUri = _messages.StringField(5)
    name = _messages.StringField(6)
    provider = _messages.EnumField('ProviderValueValuesEnum', 7)
    resourceContainer = _messages.StringField(8)
    resourceName = _messages.StringField(9)
    signalClass = _messages.EnumField('SignalClassValueValuesEnum', 10)
    signalId = _messages.StringField(11)
    signalType = _messages.EnumField('SignalTypeValueValuesEnum', 12)
    state = _messages.EnumField('StateValueValuesEnum', 13)