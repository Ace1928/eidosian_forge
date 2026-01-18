from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ImageImportJob(_messages.Message):
    """ImageImportJob describes the progress and result of an image import.

  Enums:
    StateValueValuesEnum: Output only. The state of the image import.

  Fields:
    cloudStorageUri: Output only. The path to the Cloud Storage file from
      which the image should be imported.
    createTime: Output only. The time the image import was created (as an API
      call, not when it was actually created in the target).
    createdResources: Output only. The resource paths of the resources created
      by the image import job.
    diskImageTargetDetails: Output only. Target details used to import a disk
      image.
    endTime: Output only. The time the image import was ended.
    errors: Output only. Provides details on the error that led to the image
      import state in case of an error.
    name: Output only. The resource path of the ImageImportJob.
    state: Output only. The state of the image import.
    steps: Output only. The image import steps list representing its progress.
    warnings: Output only. Warnings that occurred during the image import.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The state of the image import.

    Values:
      STATE_UNSPECIFIED: The state is unknown.
      PENDING: The image import has not yet started.
      RUNNING: The image import is active and running.
      SUCCEEDED: The image import has finished successfully.
      FAILED: The image import has finished with errors.
      CANCELLING: The image import is being cancelled.
      CANCELLED: The image import was cancelled.
    """
        STATE_UNSPECIFIED = 0
        PENDING = 1
        RUNNING = 2
        SUCCEEDED = 3
        FAILED = 4
        CANCELLING = 5
        CANCELLED = 6
    cloudStorageUri = _messages.StringField(1)
    createTime = _messages.StringField(2)
    createdResources = _messages.StringField(3, repeated=True)
    diskImageTargetDetails = _messages.MessageField('DiskImageTargetDetails', 4)
    endTime = _messages.StringField(5)
    errors = _messages.MessageField('Status', 6, repeated=True)
    name = _messages.StringField(7)
    state = _messages.EnumField('StateValueValuesEnum', 8)
    steps = _messages.MessageField('ImageImportStep', 9, repeated=True)
    warnings = _messages.MessageField('MigrationWarning', 10, repeated=True)