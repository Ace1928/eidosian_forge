from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LogsPolicy(_messages.Message):
    """LogsPolicy describes how outputs from a Job's Tasks (stdout/stderr) will
  be preserved.

  Enums:
    DestinationValueValuesEnum: Where logs should be saved.

  Fields:
    cloudLoggingOption: Optional. Additional settings for Cloud Logging. It
      will only take effect when the destination of `LogsPolicy` is set to
      `CLOUD_LOGGING`.
    destination: Where logs should be saved.
    logsPath: The path to which logs are saved when the destination = PATH.
      This can be a local file path on the VM, or under the mount point of a
      Persistent Disk or Filestore, or a Cloud Storage path.
  """

    class DestinationValueValuesEnum(_messages.Enum):
        """Where logs should be saved.

    Values:
      DESTINATION_UNSPECIFIED: Logs are not preserved.
      CLOUD_LOGGING: Logs are streamed to Cloud Logging.
      PATH: Logs are saved to a file path.
    """
        DESTINATION_UNSPECIFIED = 0
        CLOUD_LOGGING = 1
        PATH = 2
    cloudLoggingOption = _messages.MessageField('CloudLoggingOption', 1)
    destination = _messages.EnumField('DestinationValueValuesEnum', 2)
    logsPath = _messages.StringField(3)