from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1HyperparameterTuningJob(_messages.Message):
    """Represents a HyperparameterTuningJob. A HyperparameterTuningJob has a
  Study specification and multiple CustomJobs with identical CustomJob
  specification.

  Enums:
    StateValueValuesEnum: Output only. The detailed state of the job.

  Messages:
    LabelsValue: The labels with user-defined metadata to organize
      HyperparameterTuningJobs. Label keys and values can be no longer than 64
      characters (Unicode codepoints), can only contain lowercase letters,
      numeric characters, underscores and dashes. International characters are
      allowed. See https://goo.gl/xmQnxf for more information and examples of
      labels.

  Fields:
    createTime: Output only. Time when the HyperparameterTuningJob was
      created.
    displayName: Required. The display name of the HyperparameterTuningJob.
      The name can be up to 128 characters long and can consist of any UTF-8
      characters.
    encryptionSpec: Customer-managed encryption key options for a
      HyperparameterTuningJob. If this is set, then all resources created by
      the HyperparameterTuningJob will be encrypted with the provided
      encryption key.
    endTime: Output only. Time when the HyperparameterTuningJob entered any of
      the following states: `JOB_STATE_SUCCEEDED`, `JOB_STATE_FAILED`,
      `JOB_STATE_CANCELLED`.
    error: Output only. Only populated when job's state is JOB_STATE_FAILED or
      JOB_STATE_CANCELLED.
    labels: The labels with user-defined metadata to organize
      HyperparameterTuningJobs. Label keys and values can be no longer than 64
      characters (Unicode codepoints), can only contain lowercase letters,
      numeric characters, underscores and dashes. International characters are
      allowed. See https://goo.gl/xmQnxf for more information and examples of
      labels.
    maxFailedTrialCount: The number of failed Trials that need to be seen
      before failing the HyperparameterTuningJob. If set to 0, Vertex AI
      decides how many Trials must fail before the whole job fails.
    maxTrialCount: Required. The desired total number of Trials.
    name: Output only. Resource name of the HyperparameterTuningJob.
    parallelTrialCount: Required. The desired number of Trials to run in
      parallel.
    startTime: Output only. Time when the HyperparameterTuningJob for the
      first time entered the `JOB_STATE_RUNNING` state.
    state: Output only. The detailed state of the job.
    studySpec: Required. Study configuration of the HyperparameterTuningJob.
    trialJobSpec: Required. The spec of a trial job. The same spec applies to
      the CustomJobs created in all the trials.
    trials: Output only. Trials of the HyperparameterTuningJob.
    updateTime: Output only. Time when the HyperparameterTuningJob was most
      recently updated.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The detailed state of the job.

    Values:
      JOB_STATE_UNSPECIFIED: The job state is unspecified.
      JOB_STATE_QUEUED: The job has been just created or resumed and
        processing has not yet begun.
      JOB_STATE_PENDING: The service is preparing to run the job.
      JOB_STATE_RUNNING: The job is in progress.
      JOB_STATE_SUCCEEDED: The job completed successfully.
      JOB_STATE_FAILED: The job failed.
      JOB_STATE_CANCELLING: The job is being cancelled. From this state the
        job may only go to either `JOB_STATE_SUCCEEDED`, `JOB_STATE_FAILED` or
        `JOB_STATE_CANCELLED`.
      JOB_STATE_CANCELLED: The job has been cancelled.
      JOB_STATE_PAUSED: The job has been stopped, and can be resumed.
      JOB_STATE_EXPIRED: The job has expired.
      JOB_STATE_UPDATING: The job is being updated. Only jobs in the `RUNNING`
        state can be updated. After updating, the job goes back to the
        `RUNNING` state.
      JOB_STATE_PARTIALLY_SUCCEEDED: The job is partially succeeded, some
        results may be missing due to errors.
    """
        JOB_STATE_UNSPECIFIED = 0
        JOB_STATE_QUEUED = 1
        JOB_STATE_PENDING = 2
        JOB_STATE_RUNNING = 3
        JOB_STATE_SUCCEEDED = 4
        JOB_STATE_FAILED = 5
        JOB_STATE_CANCELLING = 6
        JOB_STATE_CANCELLED = 7
        JOB_STATE_PAUSED = 8
        JOB_STATE_EXPIRED = 9
        JOB_STATE_UPDATING = 10
        JOB_STATE_PARTIALLY_SUCCEEDED = 11

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """The labels with user-defined metadata to organize
    HyperparameterTuningJobs. Label keys and values can be no longer than 64
    characters (Unicode codepoints), can only contain lowercase letters,
    numeric characters, underscores and dashes. International characters are
    allowed. See https://goo.gl/xmQnxf for more information and examples of
    labels.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    createTime = _messages.StringField(1)
    displayName = _messages.StringField(2)
    encryptionSpec = _messages.MessageField('GoogleCloudAiplatformV1EncryptionSpec', 3)
    endTime = _messages.StringField(4)
    error = _messages.MessageField('GoogleRpcStatus', 5)
    labels = _messages.MessageField('LabelsValue', 6)
    maxFailedTrialCount = _messages.IntegerField(7, variant=_messages.Variant.INT32)
    maxTrialCount = _messages.IntegerField(8, variant=_messages.Variant.INT32)
    name = _messages.StringField(9)
    parallelTrialCount = _messages.IntegerField(10, variant=_messages.Variant.INT32)
    startTime = _messages.StringField(11)
    state = _messages.EnumField('StateValueValuesEnum', 12)
    studySpec = _messages.MessageField('GoogleCloudAiplatformV1StudySpec', 13)
    trialJobSpec = _messages.MessageField('GoogleCloudAiplatformV1CustomJobSpec', 14)
    trials = _messages.MessageField('GoogleCloudAiplatformV1Trial', 15, repeated=True)
    updateTime = _messages.StringField(16)