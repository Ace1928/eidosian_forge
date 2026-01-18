from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1PipelineTaskDetail(_messages.Message):
    """The runtime detail of a task execution.

  Enums:
    StateValueValuesEnum: Output only. State of the task.

  Messages:
    InputsValue: Output only. The runtime input artifacts of the task.
    OutputsValue: Output only. The runtime output artifacts of the task.

  Fields:
    createTime: Output only. Task create time.
    endTime: Output only. Task end time.
    error: Output only. The error that occurred during task execution. Only
      populated when the task's state is FAILED or CANCELLED.
    execution: Output only. The execution metadata of the task.
    executorDetail: Output only. The detailed execution info.
    inputs: Output only. The runtime input artifacts of the task.
    outputs: Output only. The runtime output artifacts of the task.
    parentTaskId: Output only. The id of the parent task if the task is within
      a component scope. Empty if the task is at the root level.
    pipelineTaskStatus: Output only. A list of task status. This field keeps a
      record of task status evolving over time.
    startTime: Output only. Task start time.
    state: Output only. State of the task.
    taskId: Output only. The system generated ID of the task.
    taskName: Output only. The user specified name of the task that is defined
      in pipeline_spec.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. State of the task.

    Values:
      STATE_UNSPECIFIED: Unspecified.
      PENDING: Specifies pending state for the task.
      RUNNING: Specifies task is being executed.
      SUCCEEDED: Specifies task completed successfully.
      CANCEL_PENDING: Specifies Task cancel is in pending state.
      CANCELLING: Specifies task is being cancelled.
      CANCELLED: Specifies task was cancelled.
      FAILED: Specifies task failed.
      SKIPPED: Specifies task was skipped due to cache hit.
      NOT_TRIGGERED: Specifies that the task was not triggered because the
        task's trigger policy is not satisfied. The trigger policy is
        specified in the `condition` field of PipelineJob.pipeline_spec.
    """
        STATE_UNSPECIFIED = 0
        PENDING = 1
        RUNNING = 2
        SUCCEEDED = 3
        CANCEL_PENDING = 4
        CANCELLING = 5
        CANCELLED = 6
        FAILED = 7
        SKIPPED = 8
        NOT_TRIGGERED = 9

    @encoding.MapUnrecognizedFields('additionalProperties')
    class InputsValue(_messages.Message):
        """Output only. The runtime input artifacts of the task.

    Messages:
      AdditionalProperty: An additional property for a InputsValue object.

    Fields:
      additionalProperties: Additional properties of type InputsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a InputsValue object.

      Fields:
        key: Name of the additional property.
        value: A GoogleCloudAiplatformV1beta1PipelineTaskDetailArtifactList
          attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('GoogleCloudAiplatformV1beta1PipelineTaskDetailArtifactList', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class OutputsValue(_messages.Message):
        """Output only. The runtime output artifacts of the task.

    Messages:
      AdditionalProperty: An additional property for a OutputsValue object.

    Fields:
      additionalProperties: Additional properties of type OutputsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a OutputsValue object.

      Fields:
        key: Name of the additional property.
        value: A GoogleCloudAiplatformV1beta1PipelineTaskDetailArtifactList
          attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('GoogleCloudAiplatformV1beta1PipelineTaskDetailArtifactList', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    createTime = _messages.StringField(1)
    endTime = _messages.StringField(2)
    error = _messages.MessageField('GoogleRpcStatus', 3)
    execution = _messages.MessageField('GoogleCloudAiplatformV1beta1Execution', 4)
    executorDetail = _messages.MessageField('GoogleCloudAiplatformV1beta1PipelineTaskExecutorDetail', 5)
    inputs = _messages.MessageField('InputsValue', 6)
    outputs = _messages.MessageField('OutputsValue', 7)
    parentTaskId = _messages.IntegerField(8)
    pipelineTaskStatus = _messages.MessageField('GoogleCloudAiplatformV1beta1PipelineTaskDetailPipelineTaskStatus', 9, repeated=True)
    startTime = _messages.StringField(10)
    state = _messages.EnumField('StateValueValuesEnum', 11)
    taskId = _messages.IntegerField(12)
    taskName = _messages.StringField(13)