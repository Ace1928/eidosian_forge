from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FirewallEndpointAssociation(_messages.Message):
    """Message describing Association object

  Enums:
    StateValueValuesEnum: Output only. Current state of the association.

  Messages:
    LabelsValue: Optional. Labels as key value pairs

  Fields:
    createTime: Output only. Create time stamp
    disabled: Optional. Whether the association is disabled. True indicates
      that traffic won't be intercepted
    firewallEndpoint: Required. The URL of the FirewallEndpoint that is being
      associated.
    labels: Optional. Labels as key value pairs
    name: Immutable. Identifier. name of resource
    network: Required. The URL of the network that is being associated.
    reconciling: Output only. Whether reconciling is in progress, recommended
      per https://google.aip.dev/128.
    state: Output only. Current state of the association.
    tlsInspectionPolicy: Optional. The URL of the TlsInspectionPolicy that is
      being associated.
    updateTime: Output only. Update time stamp
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. Current state of the association.

    Values:
      STATE_UNSPECIFIED: Not set.
      CREATING: Being created.
      ACTIVE: Active and ready for traffic.
      DELETING: Being deleted.
      INACTIVE: Down or in an error state.
    """
        STATE_UNSPECIFIED = 0
        CREATING = 1
        ACTIVE = 2
        DELETING = 3
        INACTIVE = 4

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. Labels as key value pairs

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
    disabled = _messages.BooleanField(2)
    firewallEndpoint = _messages.StringField(3)
    labels = _messages.MessageField('LabelsValue', 4)
    name = _messages.StringField(5)
    network = _messages.StringField(6)
    reconciling = _messages.BooleanField(7)
    state = _messages.EnumField('StateValueValuesEnum', 8)
    tlsInspectionPolicy = _messages.StringField(9)
    updateTime = _messages.StringField(10)