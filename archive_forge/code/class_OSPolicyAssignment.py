from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OSPolicyAssignment(_messages.Message):
    """OS policy assignment is an API resource that is used to apply a set of
  OS policies to a dynamically targeted group of Compute Engine VM instances.
  An OS policy is used to define the desired state configuration for a Compute
  Engine VM instance through a set of configuration resources that provide
  capabilities such as installing or removing software packages, or executing
  a script. For more information, see [OS policy and OS policy
  assignment](https://cloud.google.com/compute/docs/os-configuration-
  management/working-with-os-policies).

  Enums:
    RolloutStateValueValuesEnum: Output only. OS policy assignment rollout
      state

  Fields:
    baseline: Output only. Indicates that this revision has been successfully
      rolled out in this zone and new VMs will be assigned OS policies from
      this revision. For a given OS policy assignment, there is only one
      revision with a value of `true` for this field.
    deleted: Output only. Indicates that this revision deletes the OS policy
      assignment.
    description: OS policy assignment description. Length of the description
      is limited to 1024 characters.
    etag: The etag for this OS policy assignment. If this is provided on
      update, it must match the server's etag.
    instanceFilter: Required. Filter to select VMs.
    name: Resource name. Format: `projects/{project_number}/locations/{locatio
      n}/osPolicyAssignments/{os_policy_assignment_id}` This field is ignored
      when you create an OS policy assignment.
    osPolicies: Required. List of OS policies to be applied to the VMs.
    reconciling: Output only. Indicates that reconciliation is in progress for
      the revision. This value is `true` when the `rollout_state` is one of: *
      IN_PROGRESS * CANCELLING
    revisionCreateTime: Output only. The timestamp that the revision was
      created.
    revisionId: Output only. The assignment revision ID A new revision is
      committed whenever a rollout is triggered for a OS policy assignment
    rollout: Required. Rollout to deploy the OS policy assignment. A rollout
      is triggered in the following situations: 1) OSPolicyAssignment is
      created. 2) OSPolicyAssignment is updated and the update contains
      changes to one of the following fields: - instance_filter - os_policies
      3) OSPolicyAssignment is deleted.
    rolloutState: Output only. OS policy assignment rollout state
    uid: Output only. Server generated unique id for the OS policy assignment
      resource.
  """

    class RolloutStateValueValuesEnum(_messages.Enum):
        """Output only. OS policy assignment rollout state

    Values:
      ROLLOUT_STATE_UNSPECIFIED: Invalid value
      IN_PROGRESS: The rollout is in progress.
      CANCELLING: The rollout is being cancelled.
      CANCELLED: The rollout is cancelled.
      SUCCEEDED: The rollout has completed successfully.
    """
        ROLLOUT_STATE_UNSPECIFIED = 0
        IN_PROGRESS = 1
        CANCELLING = 2
        CANCELLED = 3
        SUCCEEDED = 4
    baseline = _messages.BooleanField(1)
    deleted = _messages.BooleanField(2)
    description = _messages.StringField(3)
    etag = _messages.StringField(4)
    instanceFilter = _messages.MessageField('OSPolicyAssignmentInstanceFilter', 5)
    name = _messages.StringField(6)
    osPolicies = _messages.MessageField('OSPolicy', 7, repeated=True)
    reconciling = _messages.BooleanField(8)
    revisionCreateTime = _messages.StringField(9)
    revisionId = _messages.StringField(10)
    rollout = _messages.MessageField('OSPolicyAssignmentRollout', 11)
    rolloutState = _messages.EnumField('RolloutStateValueValuesEnum', 12)
    uid = _messages.StringField(13)