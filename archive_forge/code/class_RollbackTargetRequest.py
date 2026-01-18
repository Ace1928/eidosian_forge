from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RollbackTargetRequest(_messages.Message):
    """The request object for `RollbackTarget`.

  Fields:
    overrideDeployPolicy: Optional. Deploy policies to override. Format is
      `projects/{project}/locations/{location}/deployPolicies/{deploy_policy}`
      .
    releaseId: Optional. ID of the `Release` to roll back to. If this isn't
      specified, the previous successful `Rollout` to the specified target
      will be used to determine the `Release`.
    rollbackConfig: Optional. Configs for the rollback `Rollout`.
    rolloutId: Required. ID of the rollback `Rollout` to create.
    rolloutToRollBack: Optional. If provided, this must be the latest
      `Rollout` that is on the `Target`.
    targetId: Required. ID of the `Target` that is being rolled back.
    validateOnly: Optional. If set to true, the request is validated and the
      user is provided with a `RollbackTargetResponse`.
  """
    overrideDeployPolicy = _messages.StringField(1, repeated=True)
    releaseId = _messages.StringField(2)
    rollbackConfig = _messages.MessageField('RollbackTargetConfig', 3)
    rolloutId = _messages.StringField(4)
    rolloutToRollBack = _messages.StringField(5)
    targetId = _messages.StringField(6)
    validateOnly = _messages.BooleanField(7)