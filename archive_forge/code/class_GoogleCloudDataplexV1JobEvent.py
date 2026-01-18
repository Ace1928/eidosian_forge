from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1JobEvent(_messages.Message):
    """The payload associated with Job logs that contains events describing
  jobs that have run within a Lake.

  Enums:
    ExecutionTriggerValueValuesEnum: Job execution trigger.
    ServiceValueValuesEnum: The service used to execute the job.
    StateValueValuesEnum: The job state on completion.
    TypeValueValuesEnum: The type of the job.

  Fields:
    endTime: The time when the job ended running.
    executionTrigger: Job execution trigger.
    jobId: The unique id identifying the job.
    message: The log message.
    retries: The number of retries.
    service: The service used to execute the job.
    serviceJob: The reference to the job within the service.
    startTime: The time when the job started running.
    state: The job state on completion.
    type: The type of the job.
  """

    class ExecutionTriggerValueValuesEnum(_messages.Enum):
        """Job execution trigger.

    Values:
      EXECUTION_TRIGGER_UNSPECIFIED: The job execution trigger is unspecified.
      TASK_CONFIG: The job was triggered by Dataplex based on trigger spec
        from task definition.
      RUN_REQUEST: The job was triggered by the explicit call of Task API.
    """
        EXECUTION_TRIGGER_UNSPECIFIED = 0
        TASK_CONFIG = 1
        RUN_REQUEST = 2

    class ServiceValueValuesEnum(_messages.Enum):
        """The service used to execute the job.

    Values:
      SERVICE_UNSPECIFIED: Unspecified service.
      DATAPROC: Cloud Dataproc.
    """
        SERVICE_UNSPECIFIED = 0
        DATAPROC = 1

    class StateValueValuesEnum(_messages.Enum):
        """The job state on completion.

    Values:
      STATE_UNSPECIFIED: Unspecified job state.
      SUCCEEDED: Job successfully completed.
      FAILED: Job was unsuccessful.
      CANCELLED: Job was cancelled by the user.
      ABORTED: Job was cancelled or aborted via the service executing the job.
    """
        STATE_UNSPECIFIED = 0
        SUCCEEDED = 1
        FAILED = 2
        CANCELLED = 3
        ABORTED = 4

    class TypeValueValuesEnum(_messages.Enum):
        """The type of the job.

    Values:
      TYPE_UNSPECIFIED: Unspecified job type.
      SPARK: Spark jobs.
      NOTEBOOK: Notebook jobs.
    """
        TYPE_UNSPECIFIED = 0
        SPARK = 1
        NOTEBOOK = 2
    endTime = _messages.StringField(1)
    executionTrigger = _messages.EnumField('ExecutionTriggerValueValuesEnum', 2)
    jobId = _messages.StringField(3)
    message = _messages.StringField(4)
    retries = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    service = _messages.EnumField('ServiceValueValuesEnum', 6)
    serviceJob = _messages.StringField(7)
    startTime = _messages.StringField(8)
    state = _messages.EnumField('StateValueValuesEnum', 9)
    type = _messages.EnumField('TypeValueValuesEnum', 10)