from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExecuteAirflowCommandResponse(_messages.Message):
    """Response to ExecuteAirflowCommandRequest.

  Fields:
    error: Error message. Empty if there was no error.
    executionId: The unique ID of the command execution for polling.
    pod: The name of the pod where the command is executed.
    podNamespace: The namespace of the pod where the command is executed.
  """
    error = _messages.StringField(1)
    executionId = _messages.StringField(2)
    pod = _messages.StringField(3)
    podNamespace = _messages.StringField(4)