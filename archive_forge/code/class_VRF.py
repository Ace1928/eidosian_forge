from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VRF(_messages.Message):
    """A network VRF.

  Enums:
    StateValueValuesEnum: The possible state of VRF.

  Fields:
    name: The name of the VRF.
    qosPolicy: The QOS policy applied to this VRF. The value is only
      meaningful when all the vlan attachments have the same QoS. This field
      should not be used for new integrations, use vlan attachment level qos
      instead. The field is left for backward-compatibility.
    state: The possible state of VRF.
    vlanAttachments: The list of VLAN attachments for the VRF.
  """

    class StateValueValuesEnum(_messages.Enum):
        """The possible state of VRF.

    Values:
      STATE_UNSPECIFIED: The unspecified state.
      PROVISIONING: The vrf is provisioning.
      PROVISIONED: The vrf is provisioned.
    """
        STATE_UNSPECIFIED = 0
        PROVISIONING = 1
        PROVISIONED = 2
    name = _messages.StringField(1)
    qosPolicy = _messages.MessageField('QosPolicy', 2)
    state = _messages.EnumField('StateValueValuesEnum', 3)
    vlanAttachments = _messages.MessageField('VlanAttachment', 4, repeated=True)