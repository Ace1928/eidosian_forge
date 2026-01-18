from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PatchDeployment(_messages.Message):
    """Patch deployments are configurations that individual patch jobs use to
  complete a patch. These configurations include instance filter, package
  repository settings, and a schedule. For more information about creating and
  managing patch deployments, see [Scheduling patch
  jobs](https://cloud.google.com/compute/docs/os-patch-management/schedule-
  patch-jobs).

  Enums:
    StateValueValuesEnum: Output only. Current state of the patch deployment.

  Fields:
    createTime: Output only. Time the patch deployment was created. Timestamp
      is in [RFC3339](https://www.ietf.org/rfc/rfc3339.txt) text format.
    description: Optional. Description of the patch deployment. Length of the
      description is limited to 1024 characters.
    duration: Optional. Duration of the patch. After the duration ends, the
      patch times out.
    instanceFilter: Required. VM instances to patch.
    lastExecuteTime: Output only. The last time a patch job was started by
      this deployment. Timestamp is in
      [RFC3339](https://www.ietf.org/rfc/rfc3339.txt) text format.
    name: Unique name for the patch deployment resource in a project. The
      patch deployment name is in the form:
      `projects/{project_id}/patchDeployments/{patch_deployment_id}`. This
      field is ignored when you create a new patch deployment.
    oneTimeSchedule: Required. Schedule a one-time execution.
    patchConfig: Optional. Patch configuration that is applied.
    recurringSchedule: Required. Schedule recurring executions.
    rollout: Optional. Rollout strategy of the patch job.
    state: Output only. Current state of the patch deployment.
    updateTime: Output only. Time the patch deployment was last updated.
      Timestamp is in [RFC3339](https://www.ietf.org/rfc/rfc3339.txt) text
      format.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. Current state of the patch deployment.

    Values:
      STATE_UNSPECIFIED: The default value. This value is used if the state is
        omitted.
      ACTIVE: Active value means that patch deployment generates Patch Jobs.
      PAUSED: Paused value means that patch deployment does not generate Patch
        jobs. Requires user action to move in and out from this state.
    """
        STATE_UNSPECIFIED = 0
        ACTIVE = 1
        PAUSED = 2
    createTime = _messages.StringField(1)
    description = _messages.StringField(2)
    duration = _messages.StringField(3)
    instanceFilter = _messages.MessageField('PatchInstanceFilter', 4)
    lastExecuteTime = _messages.StringField(5)
    name = _messages.StringField(6)
    oneTimeSchedule = _messages.MessageField('OneTimeSchedule', 7)
    patchConfig = _messages.MessageField('PatchConfig', 8)
    recurringSchedule = _messages.MessageField('RecurringSchedule', 9)
    rollout = _messages.MessageField('PatchRollout', 10)
    state = _messages.EnumField('StateValueValuesEnum', 11)
    updateTime = _messages.StringField(12)