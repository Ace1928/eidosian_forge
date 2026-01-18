from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AutomationRule(_messages.Message):
    """`AutomationRule` defines the automation activities.

  Fields:
    advanceRolloutRule: Optional. The `AdvanceRolloutRule` will automatically
      advance a successful Rollout.
    promoteReleaseRule: Optional. `PromoteReleaseRule` will automatically
      promote a release from the current target to a specified target.
    repairRolloutRule: Optional. The `RepairRolloutRule` will automatically
      repair a failed rollout.
  """
    advanceRolloutRule = _messages.MessageField('AdvanceRolloutRule', 1)
    promoteReleaseRule = _messages.MessageField('PromoteReleaseRule', 2)
    repairRolloutRule = _messages.MessageField('RepairRolloutRule', 3)