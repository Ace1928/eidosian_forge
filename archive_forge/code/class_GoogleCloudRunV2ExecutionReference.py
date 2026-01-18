from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRunV2ExecutionReference(_messages.Message):
    """Reference to an Execution. Use /Executions.GetExecution with the given
  name to get full execution including the latest status.

  Fields:
    completionTime: Creation timestamp of the execution.
    createTime: Creation timestamp of the execution.
    name: Name of the execution.
  """
    completionTime = _messages.StringField(1)
    createTime = _messages.StringField(2)
    name = _messages.StringField(3)