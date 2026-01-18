from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LoggingValueValuesEnum(_messages.Enum):
    """Option to specify the logging mode, which determines if and where
    build logs are stored.

    Values:
      LOGGING_UNSPECIFIED: The service determines the logging mode. The
        default is `LEGACY`. Do not rely on the default logging behavior as it
        may change in the future.
      LEGACY: Build logs are stored in Cloud Logging and Cloud Storage.
      GCS_ONLY: Build logs are stored in Cloud Storage.
      STACKDRIVER_ONLY: This option is the same as CLOUD_LOGGING_ONLY.
      CLOUD_LOGGING_ONLY: Build logs are stored in Cloud Logging. Selecting
        this option will not allow [logs
        streaming](https://cloud.google.com/sdk/gcloud/reference/builds/log).
      NONE: Turn off all logging. No build logs will be captured.
    """
    LOGGING_UNSPECIFIED = 0
    LEGACY = 1
    GCS_ONLY = 2
    STACKDRIVER_ONLY = 3
    CLOUD_LOGGING_ONLY = 4
    NONE = 5