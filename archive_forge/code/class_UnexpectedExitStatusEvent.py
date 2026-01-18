from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UnexpectedExitStatusEvent(_messages.Message):
    """An event generated when the execution of a container results in a non-
  zero exit status that was not otherwise ignored. Execution will continue,
  but only actions that are flagged as `ALWAYS_RUN` will be executed. Other
  actions will be skipped.

  Fields:
    actionId: The numeric ID of the action that started the container.
    exitStatus: The exit status of the container.
  """
    actionId = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    exitStatus = _messages.IntegerField(2, variant=_messages.Variant.INT32)