from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PromoteReleaseOperation(_messages.Message):
    """Contains the information of an automated promote-release operation.

  Fields:
    phase: Output only. The starting phase of the rollout created by this
      operation.
    rollout: Output only. The name of the rollout that initiates the
      `AutomationRun`.
    targetId: Output only. The ID of the target that represents the promotion
      stage to which the release will be promoted. The value of this field is
      the last segment of a target name.
    wait: Output only. How long the operation will be paused.
  """
    phase = _messages.StringField(1)
    rollout = _messages.StringField(2)
    targetId = _messages.StringField(3)
    wait = _messages.StringField(4)