from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ActionCondition(_messages.Message):
    """Conditions for actions to deal with task failures.

  Fields:
    exitCodes: Exit codes of a task execution. If there are more than 1 exit
      codes, when task executes with any of the exit code in the list, the
      condition is met and the action will be executed.
  """
    exitCodes = _messages.IntegerField(1, repeated=True, variant=_messages.Variant.INT32)