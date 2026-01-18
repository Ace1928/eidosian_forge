from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RepairRolloutRule(_messages.Message):
    """The `RepairRolloutRule` automation rule will automatically repair a
  failed `Rollout`.

  Enums:
    WaitPolicyValueValuesEnum: Optional. WaitForDeployPolicy delays a
      `Rollout` repair when a deploy policy violation is encountered.

  Fields:
    condition: Output only. Information around the state of the 'Automation'
      rule.
    id: Required. ID of the rule. This id must be unique in the `Automation`
      resource to which this rule belongs. The format is `a-z{0,62}`.
    jobs: Optional. Jobs to repair. Proceeds only after job name matched any
      one in the list, or for all jobs if unspecified or empty. The phase that
      includes the job must match the phase ID specified in `source_phase`.
      This value must consist of lower-case letters, numbers, and hyphens,
      start with a letter and end with a letter or a number, and have a max
      length of 63 characters. In other words, it must match the following
      regex: `^[a-z]([a-z0-9-]{0,61}[a-z0-9])?$`.
    repairModes: Required. Defines the types of automatic repair actions for
      failed jobs.
    sourcePhases: Optional. Phases within which jobs are subject to automatic
      repair actions on failure. Proceeds only after phase name matched any
      one in the list, or for all phases if unspecified. This value must
      consist of lower-case letters, numbers, and hyphens, start with a letter
      and end with a letter or a number, and have a max length of 63
      characters. In other words, it must match the following regex:
      `^[a-z]([a-z0-9-]{0,61}[a-z0-9])?$`.
    waitPolicy: Optional. WaitForDeployPolicy delays a `Rollout` repair when a
      deploy policy violation is encountered.
  """

    class WaitPolicyValueValuesEnum(_messages.Enum):
        """Optional. WaitForDeployPolicy delays a `Rollout` repair when a deploy
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
    id = _messages.StringField(2)
    jobs = _messages.StringField(3, repeated=True)
    repairModes = _messages.MessageField('RepairMode', 4, repeated=True)
    sourcePhases = _messages.StringField(5, repeated=True)
    waitPolicy = _messages.EnumField('WaitPolicyValueValuesEnum', 6)