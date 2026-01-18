from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OSPolicy(_messages.Message):
    """An OS policy defines the desired state configuration for a VM.

  Enums:
    ModeValueValuesEnum: Required. Policy mode

  Fields:
    allowNoResourceGroupMatch: This flag determines the OS policy compliance
      status when none of the resource groups within the policy are applicable
      for a VM. Set this value to `true` if the policy needs to be reported as
      compliant even if the policy has nothing to validate or enforce.
    description: Policy description. Length of the description is limited to
      1024 characters.
    id: Required. The id of the OS policy with the following restrictions: *
      Must contain only lowercase letters, numbers, and hyphens. * Must start
      with a letter. * Must be between 1-63 characters. * Must end with a
      number or a letter. * Must be unique within the assignment.
    mode: Required. Policy mode
    resourceGroups: Required. List of resource groups for the policy. For a
      particular VM, resource groups are evaluated in the order specified and
      the first resource group that is applicable is selected and the rest are
      ignored. If none of the resource groups are applicable for a VM, the VM
      is considered to be non-compliant w.r.t this policy. This behavior can
      be toggled by the flag `allow_no_resource_group_match`
  """

    class ModeValueValuesEnum(_messages.Enum):
        """Required. Policy mode

    Values:
      MODE_UNSPECIFIED: Invalid mode
      VALIDATION: This mode checks if the configuration resources in the
        policy are in their desired state. No actions are performed if they
        are not in the desired state. This mode is used for reporting
        purposes.
      ENFORCEMENT: This mode checks if the configuration resources in the
        policy are in their desired state, and if not, enforces the desired
        state.
    """
        MODE_UNSPECIFIED = 0
        VALIDATION = 1
        ENFORCEMENT = 2
    allowNoResourceGroupMatch = _messages.BooleanField(1)
    description = _messages.StringField(2)
    id = _messages.StringField(3)
    mode = _messages.EnumField('ModeValueValuesEnum', 4)
    resourceGroups = _messages.MessageField('OSPolicyResourceGroup', 5, repeated=True)