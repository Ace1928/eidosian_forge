from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1Task(_messages.Message):
    """A task represents a user-visible job.

  Enums:
    StateValueValuesEnum: Output only. Current state of the task.

  Messages:
    LabelsValue: Optional. User-defined labels for the task.

  Fields:
    createTime: Output only. The time when the task was created.
    description: Optional. Description of the task.
    displayName: Optional. User friendly display name.
    executionSpec: Required. Spec related to how a task is executed.
    executionStatus: Output only. Status of the latest task executions.
    labels: Optional. User-defined labels for the task.
    name: Output only. The relative resource name of the task, of the form:
      projects/{project_number}/locations/{location_id}/lakes/{lake_id}/
      tasks/{task_id}.
    notebook: Config related to running scheduled Notebooks.
    spark: Config related to running custom Spark tasks.
    state: Output only. Current state of the task.
    triggerSpec: Required. Spec related to how often and when a task should be
      triggered.
    uid: Output only. System generated globally unique ID for the task. This
      ID will be different if the task is deleted and re-created with the same
      name.
    updateTime: Output only. The time when the task was last updated.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. Current state of the task.

    Values:
      STATE_UNSPECIFIED: State is not specified.
      ACTIVE: Resource is active, i.e., ready to use.
      CREATING: Resource is under creation.
      DELETING: Resource is under deletion.
      ACTION_REQUIRED: Resource is active but has unresolved actions.
    """
        STATE_UNSPECIFIED = 0
        ACTIVE = 1
        CREATING = 2
        DELETING = 3
        ACTION_REQUIRED = 4

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. User-defined labels for the task.

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
    description = _messages.StringField(2)
    displayName = _messages.StringField(3)
    executionSpec = _messages.MessageField('GoogleCloudDataplexV1TaskExecutionSpec', 4)
    executionStatus = _messages.MessageField('GoogleCloudDataplexV1TaskExecutionStatus', 5)
    labels = _messages.MessageField('LabelsValue', 6)
    name = _messages.StringField(7)
    notebook = _messages.MessageField('GoogleCloudDataplexV1TaskNotebookTaskConfig', 8)
    spark = _messages.MessageField('GoogleCloudDataplexV1TaskSparkTaskConfig', 9)
    state = _messages.EnumField('StateValueValuesEnum', 10)
    triggerSpec = _messages.MessageField('GoogleCloudDataplexV1TaskTriggerSpec', 11)
    uid = _messages.StringField(12)
    updateTime = _messages.StringField(13)