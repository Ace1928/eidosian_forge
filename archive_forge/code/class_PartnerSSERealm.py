from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PartnerSSERealm(_messages.Message):
    """Message describing PartnerSSERealm object

  Enums:
    StateValueValuesEnum: Output only. [Output Only] State of the realm. It
      can be either ATTACHED or UNATTACHED

  Messages:
    LabelsValue: Labels as key value pairs

  Fields:
    createTime: Output only. [Output only] Create time stamp
    labels: Labels as key value pairs
    name: name of resource
    pairingKey: Required. value of the key to establish global handshake from
      SSERealm
    partnerNetwork: Optional. Partner-owned network to be peered with CDEN's
      sse_network in sse_project
    partnerVpc: Optional. VPC owned by the partner to be peered with CDEN
      sse_vpc in sse_project This field is deprecated. Use partner_network
      instead.
    sseNetwork: Output only. [Output only] CDEN-owned network to be peered
      with partner_network
    sseProject: Output only. [Output only] CDEN owned project owning sse_vpc
    sseVpc: Output only. [Output only] CDEN owned VPC to be peered with
      partner_vpc This field is deprecated. Use sse_network instead.
    state: Output only. [Output Only] State of the realm. It can be either
      ATTACHED or UNATTACHED
    updateTime: Output only. [Output only] Update time stamp
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. [Output Only] State of the realm. It can be either
    ATTACHED or UNATTACHED

    Values:
      STATE_UNSPECIFIED: The default value. This value is used if the state is
        omitted.
      ATTACHED: This PartnerSSERealm is attached to a SSERealm. This is the
        default state when a PartnerSSERealm is successfully created.
      UNATTACHED: This PartnerSSERealm is not attached to a SSERealm. This is
        the state when the paired SSERealm is deleted.
    """
        STATE_UNSPECIFIED = 0
        ATTACHED = 1
        UNATTACHED = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Labels as key value pairs

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    createTime = _messages.StringField(1)
    labels = _messages.MessageField('LabelsValue', 2)
    name = _messages.StringField(3)
    pairingKey = _messages.StringField(4)
    partnerNetwork = _messages.StringField(5)
    partnerVpc = _messages.StringField(6)
    sseNetwork = _messages.StringField(7)
    sseProject = _messages.StringField(8)
    sseVpc = _messages.StringField(9)
    state = _messages.EnumField('StateValueValuesEnum', 10)
    updateTime = _messages.StringField(11)