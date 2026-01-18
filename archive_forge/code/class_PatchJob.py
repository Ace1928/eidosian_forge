from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PatchJob(_messages.Message):
    """A high level representation of a patch job that is either in progress or
  has completed. Instance details are not included in the job. To paginate
  through instance details, use `ListPatchJobInstanceDetails`. For more
  information about patch jobs, see [Creating patch
  jobs](https://cloud.google.com/compute/docs/os-patch-management/create-
  patch-job).

  Enums:
    StateValueValuesEnum: The current state of the PatchJob.

  Fields:
    createTime: Time this patch job was created.
    description: Description of the patch job. Length of the description is
      limited to 1024 characters.
    displayName: Display name for this patch job. This is not a unique
      identifier.
    dryRun: If this patch job is a dry run, the agent reports that it has
      finished without running any updates on the VM instance.
    duration: Duration of the patch job. After the duration ends, the patch
      job times out.
    errorMessage: If this patch job failed, this message provides information
      about the failure.
    instanceDetailsSummary: Summary of instance details.
    instanceFilter: Instances to patch.
    name: Unique identifier for this patch job in the form
      `projects/*/patchJobs/*`
    patchConfig: Patch configuration being applied.
    patchDeployment: Output only. Name of the patch deployment that created
      this patch job.
    percentComplete: Reflects the overall progress of the patch job in the
      range of 0.0 being no progress to 100.0 being complete.
    rollout: Rollout strategy being applied.
    state: The current state of the PatchJob.
    updateTime: Last time this patch job was updated.
  """

    class StateValueValuesEnum(_messages.Enum):
        """The current state of the PatchJob.

    Values:
      STATE_UNSPECIFIED: State must be specified.
      STARTED: The patch job was successfully initiated.
      INSTANCE_LOOKUP: The patch job is looking up instances to run the patch
        on.
      PATCHING: Instances are being patched.
      SUCCEEDED: Patch job completed successfully.
      COMPLETED_WITH_ERRORS: Patch job completed but there were errors.
      CANCELED: The patch job was canceled.
      TIMED_OUT: The patch job timed out.
    """
        STATE_UNSPECIFIED = 0
        STARTED = 1
        INSTANCE_LOOKUP = 2
        PATCHING = 3
        SUCCEEDED = 4
        COMPLETED_WITH_ERRORS = 5
        CANCELED = 6
        TIMED_OUT = 7
    createTime = _messages.StringField(1)
    description = _messages.StringField(2)
    displayName = _messages.StringField(3)
    dryRun = _messages.BooleanField(4)
    duration = _messages.StringField(5)
    errorMessage = _messages.StringField(6)
    instanceDetailsSummary = _messages.MessageField('PatchJobInstanceDetailsSummary', 7)
    instanceFilter = _messages.MessageField('PatchInstanceFilter', 8)
    name = _messages.StringField(9)
    patchConfig = _messages.MessageField('PatchConfig', 10)
    patchDeployment = _messages.StringField(11)
    percentComplete = _messages.FloatField(12)
    rollout = _messages.MessageField('PatchRollout', 13)
    state = _messages.EnumField('StateValueValuesEnum', 14)
    updateTime = _messages.StringField(15)