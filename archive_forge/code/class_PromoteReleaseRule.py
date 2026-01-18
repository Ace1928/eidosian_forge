from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PromoteReleaseRule(_messages.Message):
    """`PromoteRelease` rule will automatically promote a release from the
  current target to a specified target.

  Enums:
    WaitPolicyValueValuesEnum: Optional. WaitForDeployPolicy delays a release
      promotion when a deploy policy violation is encountered.

  Fields:
    condition: Output only. Information around the state of the Automation
      rule.
    destinationPhase: Optional. The starting phase of the rollout created by
      this operation. Default to the first phase.
    destinationTargetId: Optional. The ID of the stage in the pipeline to
      which this `Release` is deploying. If unspecified, default it to the
      next stage in the promotion flow. The value of this field could be one
      of the following: * The last segment of a target name. It only needs the
      ID to determine if the target is one of the stages in the promotion
      sequence defined in the pipeline. * "@next", the next target in the
      promotion sequence.
    id: Required. ID of the rule. This id must be unique in the `Automation`
      resource to which this rule belongs. The format is `a-z{0,62}`.
    wait: Optional. How long the release need to be paused until being
      promoted to the next target.
    waitPolicy: Optional. WaitForDeployPolicy delays a release promotion when
      a deploy policy violation is encountered.
  """

    class WaitPolicyValueValuesEnum(_messages.Enum):
        """Optional. WaitForDeployPolicy delays a release promotion when a deploy
    policy violation is encountered.

    Values:
      WAIT_FOR_DEPLOY_POLICY_UNSPECIFIED: No WaitForDeployPolicy is specified.
      NEVER: Never waits on DeployPolicy, terminates `AutomationRun` if
        DeployPolicy check failed.
      LATEST: When policy passes, execute the latest `AutomationRun` only.
    """
        WAIT_FOR_DEPLOY_POLICY_UNSPECIFIED = 0
        NEVER = 1
        LATEST = 2
    condition = _messages.MessageField('AutomationRuleCondition', 1)
    destinationPhase = _messages.StringField(2)
    destinationTargetId = _messages.StringField(3)
    id = _messages.StringField(4)
    wait = _messages.StringField(5)
    waitPolicy = _messages.EnumField('WaitPolicyValueValuesEnum', 6)