from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class JobRun(_messages.Message):
    """A `JobRun` resource in the Cloud Deploy API. A `JobRun` contains
  information of a single `Rollout` job evaluation.

  Enums:
    StateValueValuesEnum: Output only. The current state of the `JobRun`.

  Fields:
    advanceChildRolloutJobRun: Output only. Information specific to an
      advanceChildRollout `JobRun`
    createChildRolloutJobRun: Output only. Information specific to a
      createChildRollout `JobRun`.
    createTime: Output only. Time at which the `JobRun` was created.
    deployJobRun: Output only. Information specific to a deploy `JobRun`.
    endTime: Output only. Time at which the `JobRun` ended.
    etag: Output only. This checksum is computed by the server based on the
      value of other fields, and may be sent on update and delete requests to
      ensure the client has an up-to-date value before proceeding.
    jobId: Output only. ID of the `Rollout` job this `JobRun` corresponds to.
    name: Optional. Name of the `JobRun`. Format is `projects/{project}/locati
      ons/{location}/deliveryPipelines/{deliveryPipeline}/releases/{releases}/
      rollouts/{rollouts}/jobRuns/{uuid}`.
    phaseId: Output only. ID of the `Rollout` phase this `JobRun` belongs in.
    postdeployJobRun: Output only. Information specific to a postdeploy
      `JobRun`.
    predeployJobRun: Output only. Information specific to a predeploy
      `JobRun`.
    startTime: Output only. Time at which the `JobRun` was started.
    state: Output only. The current state of the `JobRun`.
    uid: Output only. Unique identifier of the `JobRun`.
    verifyJobRun: Output only. Information specific to a verify `JobRun`.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The current state of the `JobRun`.

    Values:
      STATE_UNSPECIFIED: The `JobRun` has an unspecified state.
      IN_PROGRESS: The `JobRun` is in progress.
      SUCCEEDED: The `JobRun` has succeeded.
      FAILED: The `JobRun` has failed.
      TERMINATING: The `JobRun` is terminating.
      TERMINATED: The `JobRun` was terminated.
    """
        STATE_UNSPECIFIED = 0
        IN_PROGRESS = 1
        SUCCEEDED = 2
        FAILED = 3
        TERMINATING = 4
        TERMINATED = 5
    advanceChildRolloutJobRun = _messages.MessageField('AdvanceChildRolloutJobRun', 1)
    createChildRolloutJobRun = _messages.MessageField('CreateChildRolloutJobRun', 2)
    createTime = _messages.StringField(3)
    deployJobRun = _messages.MessageField('DeployJobRun', 4)
    endTime = _messages.StringField(5)
    etag = _messages.StringField(6)
    jobId = _messages.StringField(7)
    name = _messages.StringField(8)
    phaseId = _messages.StringField(9)
    postdeployJobRun = _messages.MessageField('PostdeployJobRun', 10)
    predeployJobRun = _messages.MessageField('PredeployJobRun', 11)
    startTime = _messages.StringField(12)
    state = _messages.EnumField('StateValueValuesEnum', 13)
    uid = _messages.StringField(14)
    verifyJobRun = _messages.MessageField('VerifyJobRun', 15)