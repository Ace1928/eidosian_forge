from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudSqlSettings(_messages.Message):
    """Settings for creating a Cloud SQL database instance.

  Enums:
    ActivationPolicyValueValuesEnum: The activation policy specifies when the
      instance is activated; it is applicable only when the instance state is
      'RUNNABLE'. Valid values: 'ALWAYS': The instance is on, and remains so
      even in the absence of connection requests. `NEVER`: The instance is
      off; it is not activated, even if a connection request arrives.
    DataDiskTypeValueValuesEnum: The type of storage: `PD_SSD` (default) or
      `PD_HDD`.
    DatabaseVersionValueValuesEnum: The database engine type and version.

  Messages:
    DatabaseFlagsValue: The database flags passed to the Cloud SQL instance at
      startup. An object containing a list of "key": value pairs. Example: {
      "name": "wrench", "mass": "1.3kg", "count": "3" }.
    UserLabelsValue: The resource labels for a Cloud SQL instance to use to
      annotate any related underlying resources such as Compute Engine VMs. An
      object containing a list of "key": "value" pairs. Example: `{ "name":
      "wrench", "mass": "18kg", "count": "3" }`.

  Fields:
    activationPolicy: The activation policy specifies when the instance is
      activated; it is applicable only when the instance state is 'RUNNABLE'.
      Valid values: 'ALWAYS': The instance is on, and remains so even in the
      absence of connection requests. `NEVER`: The instance is off; it is not
      activated, even if a connection request arrives.
    autoStorageIncrease: [default: ON] If you enable this setting, Cloud SQL
      checks your available storage every 30 seconds. If the available storage
      falls below a threshold size, Cloud SQL automatically adds additional
      storage capacity. If the available storage repeatedly falls below the
      threshold size, Cloud SQL continues to add storage until it reaches the
      maximum of 30 TB.
    dataDiskSizeGb: The storage capacity available to the database, in GB. The
      minimum (and default) size is 10GB.
    dataDiskType: The type of storage: `PD_SSD` (default) or `PD_HDD`.
    databaseFlags: The database flags passed to the Cloud SQL instance at
      startup. An object containing a list of "key": value pairs. Example: {
      "name": "wrench", "mass": "1.3kg", "count": "3" }.
    databaseVersion: The database engine type and version.
    hasRootPassword: Output only. Indicates If this connection profile root
      password is stored.
    ipConfig: The settings for IP Management. This allows to enable or disable
      the instance IP and manage which external networks can connect to the
      instance. The IPv4 address cannot be disabled.
    rootPassword: Input only. Initial root password.
    sourceId: The Database Migration Service source connection profile ID, in
      the format: `projects/my_project_name/locations/us-
      central1/connectionProfiles/connection_profile_ID`
    storageAutoResizeLimit: The maximum size to which storage capacity can be
      automatically increased. The default value is 0, which specifies that
      there is no limit.
    tier: The tier (or machine type) for this instance, for example:
      `db-n1-standard-1` (MySQL instances). For more information, see [Cloud
      SQL Instance Settings](https://cloud.google.com/sql/docs/mysql/instance-
      settings).
    userLabels: The resource labels for a Cloud SQL instance to use to
      annotate any related underlying resources such as Compute Engine VMs. An
      object containing a list of "key": "value" pairs. Example: `{ "name":
      "wrench", "mass": "18kg", "count": "3" }`.
    zone: The Google Cloud Platform zone where your Cloud SQL database
      instance is located.
  """

    class ActivationPolicyValueValuesEnum(_messages.Enum):
        """The activation policy specifies when the instance is activated; it is
    applicable only when the instance state is 'RUNNABLE'. Valid values:
    'ALWAYS': The instance is on, and remains so even in the absence of
    connection requests. `NEVER`: The instance is off; it is not activated,
    even if a connection request arrives.

    Values:
      SQL_ACTIVATION_POLICY_UNSPECIFIED: unspecified policy.
      ALWAYS: The instance is always up and running.
      NEVER: The instance should never spin up.
    """
        SQL_ACTIVATION_POLICY_UNSPECIFIED = 0
        ALWAYS = 1
        NEVER = 2

    class DataDiskTypeValueValuesEnum(_messages.Enum):
        """The type of storage: `PD_SSD` (default) or `PD_HDD`.

    Values:
      SQL_DATA_DISK_TYPE_UNSPECIFIED: Unspecified.
      PD_SSD: SSD disk.
      PD_HDD: HDD disk.
    """
        SQL_DATA_DISK_TYPE_UNSPECIFIED = 0
        PD_SSD = 1
        PD_HDD = 2

    class DatabaseVersionValueValuesEnum(_messages.Enum):
        """The database engine type and version.

    Values:
      SQL_DATABASE_VERSION_UNSPECIFIED: Unspecified version.
      MYSQL_5_6: MySQL 5.6.
      MYSQL_5_7: MySQL 5.7.
      MYSQL_8_0: MySQL 8.0.
    """
        SQL_DATABASE_VERSION_UNSPECIFIED = 0
        MYSQL_5_6 = 1
        MYSQL_5_7 = 2
        MYSQL_8_0 = 3

    @encoding.MapUnrecognizedFields('additionalProperties')
    class DatabaseFlagsValue(_messages.Message):
        """The database flags passed to the Cloud SQL instance at startup. An
    object containing a list of "key": value pairs. Example: { "name":
    "wrench", "mass": "1.3kg", "count": "3" }.

    Messages:
      AdditionalProperty: An additional property for a DatabaseFlagsValue
        object.

    Fields:
      additionalProperties: Additional properties of type DatabaseFlagsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a DatabaseFlagsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class UserLabelsValue(_messages.Message):
        """The resource labels for a Cloud SQL instance to use to annotate any
    related underlying resources such as Compute Engine VMs. An object
    containing a list of "key": "value" pairs. Example: `{ "name": "wrench",
    "mass": "18kg", "count": "3" }`.

    Messages:
      AdditionalProperty: An additional property for a UserLabelsValue object.

    Fields:
      additionalProperties: Additional properties of type UserLabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a UserLabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    activationPolicy = _messages.EnumField('ActivationPolicyValueValuesEnum', 1)
    autoStorageIncrease = _messages.BooleanField(2)
    dataDiskSizeGb = _messages.IntegerField(3)
    dataDiskType = _messages.EnumField('DataDiskTypeValueValuesEnum', 4)
    databaseFlags = _messages.MessageField('DatabaseFlagsValue', 5)
    databaseVersion = _messages.EnumField('DatabaseVersionValueValuesEnum', 6)
    hasRootPassword = _messages.BooleanField(7)
    ipConfig = _messages.MessageField('SqlIpConfig', 8)
    rootPassword = _messages.StringField(9)
    sourceId = _messages.StringField(10)
    storageAutoResizeLimit = _messages.IntegerField(11)
    tier = _messages.StringField(12)
    userLabels = _messages.MessageField('UserLabelsValue', 13)
    zone = _messages.StringField(14)