from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RoboScriptExecution(_messages.Message):
    """Execution stats for a user-provided Robo script.

  Fields:
    successfulActions: The number of Robo script actions executed
      successfully.
    totalActions: The total number of actions in the Robo script.
  """
    successfulActions = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    totalActions = _messages.IntegerField(2, variant=_messages.Variant.INT32)