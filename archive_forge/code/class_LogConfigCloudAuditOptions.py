from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LogConfigCloudAuditOptions(_messages.Message):
    """Write a Cloud Audit log

  Enums:
    LogNameValueValuesEnum: The log_name to populate in the Cloud Audit
      Record.
    PermissionTypeValueValuesEnum: The type associated with the permission.

  Fields:
    authorizationLoggingOptions: Information used by the Cloud Audit Logging
      pipeline. Will be deprecated once the migration to PermissionType is
      complete (b/201806118).
    logName: The log_name to populate in the Cloud Audit Record.
    permissionType: The type associated with the permission.
  """

    class LogNameValueValuesEnum(_messages.Enum):
        """The log_name to populate in the Cloud Audit Record.

    Values:
      UNSPECIFIED_LOG_NAME: Default. Should not be used.
      ADMIN_ACTIVITY: Corresponds to "cloudaudit.googleapis.com/activity"
      DATA_ACCESS: Corresponds to "cloudaudit.googleapis.com/data_access"
    """
        UNSPECIFIED_LOG_NAME = 0
        ADMIN_ACTIVITY = 1
        DATA_ACCESS = 2

    class PermissionTypeValueValuesEnum(_messages.Enum):
        """The type associated with the permission.

    Values:
      PERMISSION_TYPE_UNSPECIFIED: Default. Should not be used.
      ADMIN_READ: Permissions that gate reading resource configuration or
        metadata.
      ADMIN_WRITE: Permissions that gate modification of resource
        configuration or metadata.
      DATA_READ: Permissions that gate reading user-provided data.
      DATA_WRITE: Permissions that gate writing user-provided data.
    """
        PERMISSION_TYPE_UNSPECIFIED = 0
        ADMIN_READ = 1
        ADMIN_WRITE = 2
        DATA_READ = 3
        DATA_WRITE = 4
    authorizationLoggingOptions = _messages.MessageField('AuthorizationLoggingOptions', 1)
    logName = _messages.EnumField('LogNameValueValuesEnum', 2)
    permissionType = _messages.EnumField('PermissionTypeValueValuesEnum', 3)