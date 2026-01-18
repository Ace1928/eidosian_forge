from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContainerStateTerminated(_messages.Message):
    """ContainerStateWaiting is a terminated state of a container.

  Fields:
    exitCode: Exit status from the last termination of the container.
    finishedAt: Time at which the container last terminated
    message: Message regarding the last termination of the container
    reason: Reason from the last termination of the container
    startedAt: Time at which previous execution of the container started
  """
    exitCode = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    finishedAt = _messages.StringField(2)
    message = _messages.StringField(3)
    reason = _messages.StringField(4)
    startedAt = _messages.StringField(5)