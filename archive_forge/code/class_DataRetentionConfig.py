from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataRetentionConfig(_messages.Message):
    """The configuration setting for Airflow database data retention mechanism.

  Enums:
    TaskLogsStorageModeValueValuesEnum: Optional. The mode of storage for
      Airflow workers task logs.

  Fields:
    airflowDatabaseRetentionDays: Optional. The number of days describing for
      how long to store event-based records in airflow database. If the
      retention mechanism is enabled this value must be a positive integer
      otherwise, value should be set to 0.
    airflowMetadataRetentionConfig: Optional. The retention policy for airflow
      metadata database.
    taskLogsRetentionConfig: Optional. The configuration settings for task
      logs retention
    taskLogsRetentionDays: Optional. The number of days to retain task logs in
      the Cloud Logging bucket.
    taskLogsStorageMode: Optional. The mode of storage for Airflow workers
      task logs.
  """

    class TaskLogsStorageModeValueValuesEnum(_messages.Enum):
        """Optional. The mode of storage for Airflow workers task logs.

    Values:
      TASK_LOGS_STORAGE_MODE_UNSPECIFIED: This configuration is not specified
        by the user.
      CLOUD_LOGGING_AND_CLOUD_STORAGE: Store task logs in Cloud Logging and in
        the environment's Cloud Storage bucket.
      CLOUD_LOGGING_ONLY: Store task logs in Cloud Logging only.
    """
        TASK_LOGS_STORAGE_MODE_UNSPECIFIED = 0
        CLOUD_LOGGING_AND_CLOUD_STORAGE = 1
        CLOUD_LOGGING_ONLY = 2
    airflowDatabaseRetentionDays = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    airflowMetadataRetentionConfig = _messages.MessageField('AirflowMetadataRetentionPolicyConfig', 2)
    taskLogsRetentionConfig = _messages.MessageField('TaskLogsRetentionConfig', 3)
    taskLogsRetentionDays = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    taskLogsStorageMode = _messages.EnumField('TaskLogsStorageModeValueValuesEnum', 5)