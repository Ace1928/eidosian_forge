from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1Job(_messages.Message):
    """A job represents an instance of a task.

  Enums:
    ServiceValueValuesEnum: Output only. The underlying service running a job.
    StateValueValuesEnum: Output only. Execution state for the job.
    TriggerValueValuesEnum: Output only. Job execution trigger.

  Messages:
    LabelsValue: Output only. User-defined labels for the task.

  Fields:
    endTime: Output only. The time when the job ended.
    executionSpec: Output only. Spec related to how a task is executed.
    labels: Output only. User-defined labels for the task.
    message: Output only. Additional information about the current state.
    name: Output only. The relative resource name of the job, of the form: pro
      jects/{project_number}/locations/{location_id}/lakes/{lake_id}/tasks/{ta
      sk_id}/jobs/{job_id}.
    retryCount: Output only. The number of times the job has been retried
      (excluding the initial attempt).
    service: Output only. The underlying service running a job.
    serviceJob: Output only. The full resource name for the job run under a
      particular service.
    startTime: Output only. The time when the job was started.
    state: Output only. Execution state for the job.
    trigger: Output only. Job execution trigger.
    uid: Output only. System generated globally unique ID for the job.
  """

    class ServiceValueValuesEnum(_messages.Enum):
        """Output only. The underlying service running a job.

    Values:
      SERVICE_UNSPECIFIED: Service used to run the job is unspecified.
      DATAPROC: Dataproc service is used to run this job.
    """
        SERVICE_UNSPECIFIED = 0
        DATAPROC = 1

    class StateValueValuesEnum(_messages.Enum):
        """Output only. Execution state for the job.

    Values:
      STATE_UNSPECIFIED: The job state is unknown.
      RUNNING: The job is running.
      CANCELLING: The job is cancelling.
      CANCELLED: The job cancellation was successful.
      SUCCEEDED: The job completed successfully.
      FAILED: The job is no longer running due to an error.
      ABORTED: The job was cancelled outside of Dataplex.
    """
        STATE_UNSPECIFIED = 0
        RUNNING = 1
        CANCELLING = 2
        CANCELLED = 3
        SUCCEEDED = 4
        FAILED = 5
        ABORTED = 6

    class TriggerValueValuesEnum(_messages.Enum):
        """Output only. Job execution trigger.

    Values:
      TRIGGER_UNSPECIFIED: The trigger is unspecified.
      TASK_CONFIG: The job was triggered by Dataplex based on trigger spec
        from task definition.
      RUN_REQUEST: The job was triggered by the explicit call of Task API.
    """
        TRIGGER_UNSPECIFIED = 0
        TASK_CONFIG = 1
        RUN_REQUEST = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Output only. User-defined labels for the task.

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
    endTime = _messages.StringField(1)
    executionSpec = _messages.MessageField('GoogleCloudDataplexV1TaskExecutionSpec', 2)
    labels = _messages.MessageField('LabelsValue', 3)
    message = _messages.StringField(4)
    name = _messages.StringField(5)
    retryCount = _messages.IntegerField(6, variant=_messages.Variant.UINT32)
    service = _messages.EnumField('ServiceValueValuesEnum', 7)
    serviceJob = _messages.StringField(8)
    startTime = _messages.StringField(9)
    state = _messages.EnumField('StateValueValuesEnum', 10)
    trigger = _messages.EnumField('TriggerValueValuesEnum', 11)
    uid = _messages.StringField(12)