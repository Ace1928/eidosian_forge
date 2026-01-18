from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TaskExecution(_messages.Message):
    """This Task Execution field includes detail information for task execution
  procedures, based on StatusEvent types.

  Fields:
    exitCode: When task is completed as the status of FAILED or SUCCEEDED,
      exit code is for one task execution result, default is 0 as success.
    stderrSnippet: Optional. The tail end of any content written to standard
      error by the task execution. This field will be populated only when the
      execution failed.
  """
    exitCode = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    stderrSnippet = _messages.StringField(2)