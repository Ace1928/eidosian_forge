from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PollAirflowCommandResponse(_messages.Message):
    """Response to PollAirflowCommandRequest.

  Fields:
    exitInfo: The result exit status of the command.
    output: Output from the command execution. It may not contain the full
      output and the caller may need to poll for more lines.
    outputEnd: Whether the command execution has finished and there is no more
      output.
  """
    exitInfo = _messages.MessageField('ExitInfo', 1)
    output = _messages.MessageField('Line', 2, repeated=True)
    outputEnd = _messages.BooleanField(3)