from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FirewallEndpoint(_messages.Message):
    """Message describing Endpoint object

  Enums:
    StateValueValuesEnum: Output only. Current state of the endpoint.
    TypeValueValuesEnum: Optional. Endpoint type.

  Messages:
    LabelsValue: Optional. Labels as key value pairs

  Fields:
    associatedNetworks: Output only. List of networks that are associated with
      this endpoint in the local zone. This is a projection of the
      FirewallEndpointAssociations pointing at this endpoint. A network will
      only appear in this list after traffic routing is fully configured.
      Format: projects/{project}/global/networks/{name}.
    associations: Output only. List of FirewallEndpointAssociations that are
      associated to this endpoint. An association will only appear in this
      list after traffic routing is fully configured.
    billingProjectId: Required. Project to bill on endpoint uptime usage.
    createTime: Output only. Create time stamp
    description: Optional. Description of the firewall endpoint. Max length
      2048 characters.
    firstPartyEndpointSettings: Optional. Firewall endpoint settings for first
      party firewall endpoints.
    labels: Optional. Labels as key value pairs
    name: Immutable. Identifier. name of resource
    reconciling: Output only. Whether reconciling is in progress, recommended
      per https://google.aip.dev/128.
    state: Output only. Current state of the endpoint.
    thirdPartyEndpointSettings: Optional. Firewall endpoint settings for third
      party firewall endpoints.
    type: Optional. Endpoint type.
    updateTime: Output only. Update time stamp
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. Current state of the endpoint.

    Values:
      STATE_UNSPECIFIED: Not set.
      CREATING: Being created.
      ACTIVE: Processing configuration updates.
      DELETING: Being deleted.
      INACTIVE: Down or in an error state.
    """
        STATE_UNSPECIFIED = 0
        CREATING = 1
        ACTIVE = 2
        DELETING = 3
        INACTIVE = 4

    class TypeValueValuesEnum(_messages.Enum):
        """Optional. Endpoint type.

    Values:
      TYPE_UNSPECIFIED: Not set.
      FIRST_PARTY: First party firewall endpoint.
      THIRD_PARTY: Third party firewall endpoint.
    """
        TYPE_UNSPECIFIED = 0
        FIRST_PARTY = 1
        THIRD_PARTY = 2

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
    associatedNetworks = _messages.StringField(1, repeated=True)
    associations = _messages.MessageField('FirewallEndpointAssociationReference', 2, repeated=True)
    billingProjectId = _messages.StringField(3)
    createTime = _messages.StringField(4)
    description = _messages.StringField(5)
    firstPartyEndpointSettings = _messages.MessageField('FirstPartyEndpointSettings', 6)
    labels = _messages.MessageField('LabelsValue', 7)
    name = _messages.StringField(8)
    reconciling = _messages.BooleanField(9)
    state = _messages.EnumField('StateValueValuesEnum', 10)
    thirdPartyEndpointSettings = _messages.MessageField('ThirdPartyEndpointSettings', 11)
    type = _messages.EnumField('TypeValueValuesEnum', 12)
    updateTime = _messages.StringField(13)