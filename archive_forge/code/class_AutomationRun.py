from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AutomationRun(_messages.Message):
    """An `AutomationRun` resource in the Cloud Deploy API. An `AutomationRun`
  represents an execution instance of an automation rule.

  Enums:
    StateValueValuesEnum: Output only. Current state of the `AutomationRun`.

  Fields:
    advanceRolloutOperation: Output only. Advances a rollout to the next
      phase.
    automationId: Output only. The ID of the automation that initiated the
      operation.
    automationSnapshot: Output only. Snapshot of the Automation taken at
      AutomationRun creation time.
    createTime: Output only. Time at which the `AutomationRun` was created.
    etag: Output only. The weak etag of the `AutomationRun` resource. This
      checksum is computed by the server based on the value of other fields,
      and may be sent on update and delete requests to ensure the client has
      an up-to-date value before proceeding.
    expireTime: Output only. Time the `AutomationRun` expires. An
      `AutomationRun` expires after 14 days from its creation date.
    name: Output only. Name of the `AutomationRun`. Format is `projects/{proje
      ct}/locations/{location}/deliveryPipelines/{delivery_pipeline}/automatio
      nRuns/{automation_run}`.
    policyViolation: Output only. Contains information about what policies
      prevented the `AutomationRun` to proceed.
    promoteReleaseOperation: Output only. Promotes a release to a specified
      'Target'.
    repairRolloutOperation: Output only. Repairs a failed 'Rollout'.
    ruleId: Output only. The ID of the automation rule that initiated the
      operation.
    serviceAccount: Output only. Email address of the user-managed IAM service
      account that performs the operations against Cloud Deploy resources.
    state: Output only. Current state of the `AutomationRun`.
    stateDescription: Output only. Explains the current state of the
      `AutomationRun`. Present only when an explanation is needed.
    targetId: Output only. The ID of the target that represents the promotion
      stage that initiates the `AutomationRun`. The value of this field is the
      last segment of a target name.
    updateTime: Output only. Time at which the automationRun was updated.
    waitUntilTime: Output only. Earliest time the `AutomationRun` will attempt
      to resume. Wait-time is configured by `wait` in automation rule.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. Current state of the `AutomationRun`.

    Values:
      STATE_UNSPECIFIED: The `AutomationRun` has an unspecified state.
      SUCCEEDED: The `AutomationRun` has succeeded.
      CANCELLED: The `AutomationRun` was cancelled.
      FAILED: The `AutomationRun` has failed.
      IN_PROGRESS: The `AutomationRun` is in progress.
      PENDING: The `AutomationRun` is pending.
      ABORTED: The `AutomationRun` was aborted.
    """
        STATE_UNSPECIFIED = 0
        SUCCEEDED = 1
        CANCELLED = 2
        FAILED = 3
        IN_PROGRESS = 4
        PENDING = 5
        ABORTED = 6
    advanceRolloutOperation = _messages.MessageField('AdvanceRolloutOperation', 1)
    automationId = _messages.StringField(2)
    automationSnapshot = _messages.MessageField('Automation', 3)
    createTime = _messages.StringField(4)
    etag = _messages.StringField(5)
    expireTime = _messages.StringField(6)
    name = _messages.StringField(7)
    policyViolation = _messages.MessageField('PolicyViolation', 8)
    promoteReleaseOperation = _messages.MessageField('PromoteReleaseOperation', 9)
    repairRolloutOperation = _messages.MessageField('RepairRolloutOperation', 10)
    ruleId = _messages.StringField(11)
    serviceAccount = _messages.StringField(12)
    state = _messages.EnumField('StateValueValuesEnum', 13)
    stateDescription = _messages.StringField(14)
    targetId = _messages.StringField(15)
    updateTime = _messages.StringField(16)
    waitUntilTime = _messages.StringField(17)