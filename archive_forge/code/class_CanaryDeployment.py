from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CanaryDeployment(_messages.Message):
    """CanaryDeployment represents the canary deployment configuration

  Fields:
    percentages: Required. The percentage based deployments that will occur as
      a part of a `Rollout`. List is expected in ascending order and each
      integer n is 0 <= n < 100.
    postdeploy: Optional. Configuration for the postdeploy job of the last
      phase. If this is not configured, there will be no postdeploy job for
      this phase.
    predeploy: Optional. Configuration for the predeploy job of the first
      phase. If this is not configured, there will be no predeploy job for
      this phase.
    verify: Whether to run verify tests after each percentage deployment.
  """
    percentages = _messages.IntegerField(1, repeated=True, variant=_messages.Variant.INT32)
    postdeploy = _messages.MessageField('Postdeploy', 2)
    predeploy = _messages.MessageField('Predeploy', 3)
    verify = _messages.BooleanField(4)