from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StopAirflowCommandRequest(_messages.Message):
    """Stop Airflow Command request.

  Fields:
    executionId: The unique ID of the command execution.
    force: If true, the execution is terminated forcefully (SIGKILL). If
      false, the execution is stopped gracefully, giving it time for cleanup.
    pod: The name of the pod where the command is executed.
    podNamespace: The namespace of the pod where the command is executed.
  """
    executionId = _messages.StringField(1)
    force = _messages.BooleanField(2)
    pod = _messages.StringField(3)
    podNamespace = _messages.StringField(4)