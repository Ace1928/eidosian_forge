from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StepState(_messages.Message):
    """StepState reports the results of running a step in a Task.

  Fields:
    imageId: Image ID of the StepState.
    name: Name of the StepState.
    running: Details about a running container
    terminated: Details about a terminated container
    waiting: Details about a waiting container
  """
    imageId = _messages.StringField(1)
    name = _messages.StringField(2)
    running = _messages.MessageField('ContainerStateRunning', 3)
    terminated = _messages.MessageField('ContainerStateTerminated', 4)
    waiting = _messages.MessageField('ContainerStateWaiting', 5)