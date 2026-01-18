from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PollAirflowCommandRequest(_messages.Message):
    """Poll Airflow Command request.

  Fields:
    executionId: The unique ID of the command execution.
    nextLineNumber: Line number from which new logs should be fetched.
    pod: The name of the pod where the command is executed.
    podNamespace: The namespace of the pod where the command is executed.
  """
    executionId = _messages.StringField(1)
    nextLineNumber = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pod = _messages.StringField(3)
    podNamespace = _messages.StringField(4)