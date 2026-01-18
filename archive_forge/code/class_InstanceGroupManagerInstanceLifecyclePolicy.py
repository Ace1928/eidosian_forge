from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstanceGroupManagerInstanceLifecyclePolicy(_messages.Message):
    """A InstanceGroupManagerInstanceLifecyclePolicy object.

  Enums:
    DefaultActionOnFailureValueValuesEnum: The action that a MIG performs on a
      failed or an unhealthy VM. A VM is marked as unhealthy when the
      application running on that VM fails a health check. Valid values are -
      REPAIR (default): MIG automatically repairs a failed or an unhealthy VM
      by recreating it. For more information, see About repairing VMs in a
      MIG. - DO_NOTHING: MIG does not repair a failed or an unhealthy VM.
    ForceUpdateOnRepairValueValuesEnum: A bit indicating whether to forcefully
      apply the group's latest configuration when repairing a VM. Valid
      options are: - NO (default): If configuration updates are available,
      they are not forcefully applied during repair. Instead, configuration
      updates are applied according to the group's update policy. - YES: If
      configuration updates are available, they are applied during repair.

  Fields:
    defaultActionOnFailure: The action that a MIG performs on a failed or an
      unhealthy VM. A VM is marked as unhealthy when the application running
      on that VM fails a health check. Valid values are - REPAIR (default):
      MIG automatically repairs a failed or an unhealthy VM by recreating it.
      For more information, see About repairing VMs in a MIG. - DO_NOTHING:
      MIG does not repair a failed or an unhealthy VM.
    forceUpdateOnRepair: A bit indicating whether to forcefully apply the
      group's latest configuration when repairing a VM. Valid options are: -
      NO (default): If configuration updates are available, they are not
      forcefully applied during repair. Instead, configuration updates are
      applied according to the group's update policy. - YES: If configuration
      updates are available, they are applied during repair.
  """

    class DefaultActionOnFailureValueValuesEnum(_messages.Enum):
        """The action that a MIG performs on a failed or an unhealthy VM. A VM is
    marked as unhealthy when the application running on that VM fails a health
    check. Valid values are - REPAIR (default): MIG automatically repairs a
    failed or an unhealthy VM by recreating it. For more information, see
    About repairing VMs in a MIG. - DO_NOTHING: MIG does not repair a failed
    or an unhealthy VM.

    Values:
      DELETE: MIG deletes a failed or an unhealthy VM. Deleting the VM
        decreases the target size of the MIG.
      DO_NOTHING: MIG does not repair a failed or an unhealthy VM.
      REPAIR: (Default) MIG automatically repairs a failed or an unhealthy VM
        by recreating it. For more information, see About repairing VMs in a
        MIG.
    """
        DELETE = 0
        DO_NOTHING = 1
        REPAIR = 2

    class ForceUpdateOnRepairValueValuesEnum(_messages.Enum):
        """A bit indicating whether to forcefully apply the group's latest
    configuration when repairing a VM. Valid options are: - NO (default): If
    configuration updates are available, they are not forcefully applied
    during repair. Instead, configuration updates are applied according to the
    group's update policy. - YES: If configuration updates are available, they
    are applied during repair.

    Values:
      NO: <no description>
      YES: <no description>
    """
        NO = 0
        YES = 1
    defaultActionOnFailure = _messages.EnumField('DefaultActionOnFailureValueValuesEnum', 1)
    forceUpdateOnRepair = _messages.EnumField('ForceUpdateOnRepairValueValuesEnum', 2)