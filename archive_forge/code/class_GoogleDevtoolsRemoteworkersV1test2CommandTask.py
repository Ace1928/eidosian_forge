from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsRemoteworkersV1test2CommandTask(_messages.Message):
    """Describes a shell-style task to execute, suitable for providing as the
  Bots interface's `Lease.payload` field.

  Fields:
    expectedOutputs: The expected outputs from the task.
    inputs: The inputs to the task.
    timeouts: The timeouts of this task.
  """
    expectedOutputs = _messages.MessageField('GoogleDevtoolsRemoteworkersV1test2CommandTaskOutputs', 1)
    inputs = _messages.MessageField('GoogleDevtoolsRemoteworkersV1test2CommandTaskInputs', 2)
    timeouts = _messages.MessageField('GoogleDevtoolsRemoteworkersV1test2CommandTaskTimeouts', 3)