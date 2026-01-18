from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareComponentUpgrade(_messages.Message):
    """Component level upgrade resource object. Part of upgradeJob of a PC.

  Enums:
    ComponentTypeValueValuesEnum: Output only. Type of component.
    StateValueValuesEnum: Output only. The state of the resource.

  Fields:
    componentType: Output only. Type of component.
    createTime: Output only. The create time of the resource, when the upgrade
      on this component started.
    endTime: Output only. The ending time of the upgrade operation.
    state: Output only. The state of the resource.
    updateTime: Output only. Last update time of this resource.
  """

    class ComponentTypeValueValuesEnum(_messages.Enum):
        """Output only. Type of component.

    Values:
      VMWARE_COMPONENT_TYPE_UNSPECIFIED: The default value. This value should
        never be used.
      VCENTER: Vcenter server.
      ESXI: Esxi nodes + Transport nodes upgrade.
      NSXT_UC: Nsxt upgrade coordinator.
      NSXT_EDGE: Nsxt edges cluster.
      NSXT_MGR: Nsxt managers/management plane.
      HCX: HCX appliance.
      VSAN: VSAN cluster.
      DVS: DVS switch.
      NAMESERVER_VM: Nameserver VMs.
      KMS_VM: KMS VM used for vsan encryption.
      WITNESS_VM: Witness VM in case of stretch PC.
      NSXT: nsxt
    """
        VMWARE_COMPONENT_TYPE_UNSPECIFIED = 0
        VCENTER = 1
        ESXI = 2
        NSXT_UC = 3
        NSXT_EDGE = 4
        NSXT_MGR = 5
        HCX = 6
        VSAN = 7
        DVS = 8
        NAMESERVER_VM = 9
        KMS_VM = 10
        WITNESS_VM = 11
        NSXT = 12

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The state of the resource.

    Values:
      STATE_UNSPECIFIED: The default value. This value should never be used.
      RUNNING: Component's upgrade is in progress.
      PAUSED: The component's upgrade is paused. Will be resumed when upgrade
        job is resumed.
      SUCCEEDED: The component's upgrade is successfully completed.
      FAILED: The component's upgrade has failed. This will resume if upgrade
        job is resumed or stay as is.
      NOT_STARTED: Component's upgrade has not started yet.
      NOT_APPLICABLE: Component's upgrade is not applicable in this upgrade
        job. It will be skipped.
    """
        STATE_UNSPECIFIED = 0
        RUNNING = 1
        PAUSED = 2
        SUCCEEDED = 3
        FAILED = 4
        NOT_STARTED = 5
        NOT_APPLICABLE = 6
    componentType = _messages.EnumField('ComponentTypeValueValuesEnum', 1)
    createTime = _messages.StringField(2)
    endTime = _messages.StringField(3)
    state = _messages.EnumField('StateValueValuesEnum', 4)
    updateTime = _messages.StringField(5)