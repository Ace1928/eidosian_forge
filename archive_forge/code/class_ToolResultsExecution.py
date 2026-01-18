from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ToolResultsExecution(_messages.Message):
    """Represents a tool results execution resource. This has the results of a
  TestMatrix.

  Fields:
    executionId: Output only. A tool results execution ID.
    historyId: Output only. A tool results history ID.
    projectId: Output only. The cloud project that owns the tool results
      execution.
  """
    executionId = _messages.StringField(1)
    historyId = _messages.StringField(2)
    projectId = _messages.StringField(3)