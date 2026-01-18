from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstanceGroupManagerStandbyPolicy(_messages.Message):
    """A InstanceGroupManagerStandbyPolicy object.

  Enums:
    ModeValueValuesEnum: Defines how a MIG resumes or starts VMs from a
      standby pool when the group scales out. The default mode is `MANUAL`.

  Fields:
    initialDelaySec: Specifies the number of seconds that the MIG should wait
      to suspend or stop a VM after that VM was created. The initial delay
      gives the initialization script the time to prepare your VM for a quick
      scale out. The value of initial delay must be between 0 and 3600
      seconds. The default value is 0.
    mode: Defines how a MIG resumes or starts VMs from a standby pool when the
      group scales out. The default mode is `MANUAL`.
  """

    class ModeValueValuesEnum(_messages.Enum):
        """Defines how a MIG resumes or starts VMs from a standby pool when the
    group scales out. The default mode is `MANUAL`.

    Values:
      MANUAL: MIG does not automatically resume or start VMs in the standby
        pool when the group scales out.
      SCALE_OUT_POOL: MIG automatically resumes or starts VMs in the standby
        pool when the group scales out, and replenishes the standby pool
        afterwards.
    """
        MANUAL = 0
        SCALE_OUT_POOL = 1
    initialDelaySec = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    mode = _messages.EnumField('ModeValueValuesEnum', 2)