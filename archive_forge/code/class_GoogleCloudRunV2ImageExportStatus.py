from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRunV2ImageExportStatus(_messages.Message):
    """The status of an image export job.

  Enums:
    ExportJobStateValueValuesEnum: Output only. Has the image export job
      finished (regardless of successful or failure).

  Fields:
    exportJobState: Output only. Has the image export job finished (regardless
      of successful or failure).
    exportedImageDigest: The exported image ID as it will appear in Artifact
      Registry.
    status: The status of the export task if done.
    tag: The image tag as it will appear in Artifact Registry.
  """

    class ExportJobStateValueValuesEnum(_messages.Enum):
        """Output only. Has the image export job finished (regardless of
    successful or failure).

    Values:
      EXPORT_JOB_STATE_UNSPECIFIED: State unspecified.
      IN_PROGRESS: Job still in progress.
      FINISHED: Job finished.
    """
        EXPORT_JOB_STATE_UNSPECIFIED = 0
        IN_PROGRESS = 1
        FINISHED = 2
    exportJobState = _messages.EnumField('ExportJobStateValueValuesEnum', 1)
    exportedImageDigest = _messages.StringField(2)
    status = _messages.MessageField('UtilStatusProto', 3)
    tag = _messages.StringField(4)