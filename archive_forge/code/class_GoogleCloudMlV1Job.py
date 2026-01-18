from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1Job(_messages.Message):
    """Represents a training or prediction job.

  Enums:
    StateValueValuesEnum: Output only. The detailed state of a job.

  Messages:
    LabelsValue: Optional. One or more labels that you can add, to organize
      your jobs. Each label is a key-value pair, where both the key and the
      value are arbitrary strings that you supply. For more information, see
      the documentation on using labels.

  Fields:
    createTime: Output only. When the job was created.
    endTime: Output only. When the job processing was completed.
    errorMessage: Output only. The details of a failure or a cancellation.
    etag: `etag` is used for optimistic concurrency control as a way to help
      prevent simultaneous updates of a job from overwriting each other. It is
      strongly suggested that systems make use of the `etag` in the read-
      modify-write cycle to perform job updates in order to avoid race
      conditions: An `etag` is returned in the response to `GetJob`, and
      systems are expected to put that etag in the request to `UpdateJob` to
      ensure that their change will be applied to the same version of the job.
    explanationInput: Input parameters to create an explanation job.
    explanationOutput: The current explanation job result.
    jobId: Required. The user-specified id of the job.
    jobPosition: Output only. It's only effect when the job is in QUEUED
      state. If it's positive, it indicates the job's position in the job
      scheduler. It's 0 when the job is already scheduled.
    labels: Optional. One or more labels that you can add, to organize your
      jobs. Each label is a key-value pair, where both the key and the value
      are arbitrary strings that you supply. For more information, see the
      documentation on using labels.
    predictionInput: Input parameters to create a prediction job.
    predictionOutput: The current prediction job result.
    startTime: Output only. When the job processing was started.
    state: Output only. The detailed state of a job.
    trainingInput: Input parameters to create a training job.
    trainingOutput: The current training job result.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The detailed state of a job.

    Values:
      STATE_UNSPECIFIED: The job state is unspecified.
      QUEUED: The job has been just created and processing has not yet begun.
      PREPARING: The service is preparing to run the job.
      RUNNING: The job is in progress.
      SUCCEEDED: The job completed successfully.
      FAILED: The job failed. `error_message` should contain the details of
        the failure.
      CANCELLING: The job is being cancelled. `error_message` should describe
        the reason for the cancellation.
      CANCELLED: The job has been cancelled. `error_message` should describe
        the reason for the cancellation.
    """
        STATE_UNSPECIFIED = 0
        QUEUED = 1
        PREPARING = 2
        RUNNING = 3
        SUCCEEDED = 4
        FAILED = 5
        CANCELLING = 6
        CANCELLED = 7

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. One or more labels that you can add, to organize your jobs.
    Each label is a key-value pair, where both the key and the value are
    arbitrary strings that you supply. For more information, see the
    documentation on using labels.

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
    endTime = _messages.StringField(2)
    errorMessage = _messages.StringField(3)
    etag = _messages.BytesField(4)
    explanationInput = _messages.MessageField('GoogleCloudMlV1ExplanationInput', 5)
    explanationOutput = _messages.MessageField('GoogleCloudMlV1ExplanationOutput', 6)
    jobId = _messages.StringField(7)
    jobPosition = _messages.IntegerField(8)
    labels = _messages.MessageField('LabelsValue', 9)
    predictionInput = _messages.MessageField('GoogleCloudMlV1PredictionInput', 10)
    predictionOutput = _messages.MessageField('GoogleCloudMlV1PredictionOutput', 11)
    startTime = _messages.StringField(12)
    state = _messages.EnumField('StateValueValuesEnum', 13)
    trainingInput = _messages.MessageField('GoogleCloudMlV1TrainingInput', 14)
    trainingOutput = _messages.MessageField('GoogleCloudMlV1TrainingOutput', 15)