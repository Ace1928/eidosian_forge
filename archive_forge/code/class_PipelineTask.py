from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PipelineTask(_messages.Message):
    """PipelineTask defines a task in a Pipeline.

  Fields:
    name: Name of the task.
    params: Params is a list of parameter names and values.
    retries: Retries represents how many times this task should be retried in
      case of task failure.
    runAfter: RunAfter is the list of PipelineTask names that should be
      executed before this Task executes. (Used to force a specific ordering
      in graph execution.)
    taskRef: Reference to a specific instance of a task.
    taskSpec: Spec to instantiate this TaskRun.
    timeout: Time after which the TaskRun times out. Defaults to 1 hour.
      Specified TaskRun timeout should be less than 24h.
    whenExpressions: Conditions that need to be true for the task to run.
    workspaces: Workspaces maps workspaces from the pipeline spec to the
      workspaces declared in the Task.
  """
    name = _messages.StringField(1)
    params = _messages.MessageField('Param', 2, repeated=True)
    retries = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    runAfter = _messages.StringField(4, repeated=True)
    taskRef = _messages.MessageField('TaskRef', 5)
    taskSpec = _messages.MessageField('EmbeddedTask', 6)
    timeout = _messages.StringField(7)
    whenExpressions = _messages.MessageField('WhenExpression', 8, repeated=True)
    workspaces = _messages.MessageField('WorkspacePipelineTaskBinding', 9, repeated=True)