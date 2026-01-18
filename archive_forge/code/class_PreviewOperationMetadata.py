from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PreviewOperationMetadata(_messages.Message):
    """Ephemeral metadata content describing the state of a preview operation.

  Enums:
    StepValueValuesEnum: The current step the preview operation is running.

  Fields:
    build: Output only. Cloud Build instance UUID associated with this
      preview.
    logs: Output only. Location of preview logs in `gs://{bucket}/{object}`
      format.
    previewArtifacts: Artifacts from preview.
    step: The current step the preview operation is running.
  """

    class StepValueValuesEnum(_messages.Enum):
        """The current step the preview operation is running.

    Values:
      PREVIEW_STEP_UNSPECIFIED: Unspecified preview step.
      PREPARING_STORAGE_BUCKET: Infra Manager is creating a Google Cloud
        Storage bucket to store artifacts and metadata about the preview.
      DOWNLOADING_BLUEPRINT: Downloading the blueprint onto the Google Cloud
        Storage bucket.
      RUNNING_TF_INIT: Initializing Terraform using `terraform init`.
      RUNNING_TF_PLAN: Running `terraform plan`.
      FETCHING_DEPLOYMENT: Fetching a deployment.
      LOCKING_DEPLOYMENT: Locking a deployment.
      UNLOCKING_DEPLOYMENT: Unlocking a deployment.
      SUCCEEDED: Operation was successful.
      FAILED: Operation failed.
      VALIDATING_REPOSITORY: Validating the provided repository.
    """
        PREVIEW_STEP_UNSPECIFIED = 0
        PREPARING_STORAGE_BUCKET = 1
        DOWNLOADING_BLUEPRINT = 2
        RUNNING_TF_INIT = 3
        RUNNING_TF_PLAN = 4
        FETCHING_DEPLOYMENT = 5
        LOCKING_DEPLOYMENT = 6
        UNLOCKING_DEPLOYMENT = 7
        SUCCEEDED = 8
        FAILED = 9
        VALIDATING_REPOSITORY = 10
    build = _messages.StringField(1)
    logs = _messages.StringField(2)
    previewArtifacts = _messages.MessageField('PreviewArtifacts', 3)
    step = _messages.EnumField('StepValueValuesEnum', 4)