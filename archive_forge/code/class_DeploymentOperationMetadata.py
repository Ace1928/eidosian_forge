from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DeploymentOperationMetadata(_messages.Message):
    """Ephemeral metadata content describing the state of a deployment
  operation.

  Enums:
    StepValueValuesEnum: The current step the deployment operation is running.

  Fields:
    applyResults: Outputs and artifacts from applying a deployment.
    build: Output only. Cloud Build instance UUID associated with this
      operation.
    logs: Output only. Location of Deployment operations logs in
      `gs://{bucket}/{object}` format.
    step: The current step the deployment operation is running.
  """

    class StepValueValuesEnum(_messages.Enum):
        """The current step the deployment operation is running.

    Values:
      DEPLOYMENT_STEP_UNSPECIFIED: Unspecified deployment step
      PREPARING_STORAGE_BUCKET: Infra Manager is creating a Google Cloud
        Storage bucket to store artifacts and metadata about the deployment
        and revision
      DOWNLOADING_BLUEPRINT: Downloading the blueprint onto the Google Cloud
        Storage bucket
      RUNNING_TF_INIT: Initializing Terraform using `terraform init`
      RUNNING_TF_PLAN: Running `terraform plan`
      RUNNING_TF_APPLY: Actuating resources using Terraform using `terraform
        apply`
      RUNNING_TF_DESTROY: Destroying resources using Terraform using
        `terraform destroy`
      RUNNING_TF_VALIDATE: Validating the uploaded TF state file when
        unlocking a deployment
      UNLOCKING_DEPLOYMENT: Unlocking a deployment
      SUCCEEDED: Operation was successful
      FAILED: Operation failed
      VALIDATING_REPOSITORY: Validating the provided repository.
      RUNNING_QUOTA_VALIDATION: Running quota validation
    """
        DEPLOYMENT_STEP_UNSPECIFIED = 0
        PREPARING_STORAGE_BUCKET = 1
        DOWNLOADING_BLUEPRINT = 2
        RUNNING_TF_INIT = 3
        RUNNING_TF_PLAN = 4
        RUNNING_TF_APPLY = 5
        RUNNING_TF_DESTROY = 6
        RUNNING_TF_VALIDATE = 7
        UNLOCKING_DEPLOYMENT = 8
        SUCCEEDED = 9
        FAILED = 10
        VALIDATING_REPOSITORY = 11
        RUNNING_QUOTA_VALIDATION = 12
    applyResults = _messages.MessageField('ApplyResults', 1)
    build = _messages.StringField(2)
    logs = _messages.StringField(3)
    step = _messages.EnumField('StepValueValuesEnum', 4)