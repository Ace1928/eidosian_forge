from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Peering(_messages.Message):
    """Represents a Managed Service for Microsoft Active Directory Peering.

  Enums:
    StateValueValuesEnum: Output only. The current state of this Peering.

  Messages:
    LabelsValue: Optional. Resource labels to represent user-provided
      metadata.

  Fields:
    authorizedNetwork: Required. The full names of the Google Compute Engine
      [networks](/compute/docs/networks-and-firewalls#networks) to which the
      instance is connected. Caller needs to make sure that CIDR subnets do
      not overlap between networks, else peering creation will fail.
    createTime: Output only. The time the instance was created.
    domainResource: Required. Full domain resource path for the Managed AD
      Domain involved in peering. The resource path should be in the form:
      `projects/{project_id}/locations/global/domains/{domain_name}`
    labels: Optional. Resource labels to represent user-provided metadata.
    name: Output only. Unique name of the peering in this scope including
      projects and location using the form:
      `projects/{project_id}/locations/global/peerings/{peering_id}`.
    state: Output only. The current state of this Peering.
    statusMessage: Output only. Additional information about the current
      status of this peering, if available.
    updateTime: Output only. Last update time.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The current state of this Peering.

    Values:
      STATE_UNSPECIFIED: Not set.
      CREATING: Peering is being created.
      CONNECTED: Peering is connected.
      DISCONNECTED: Peering is disconnected.
      DELETING: Peering is being deleted.
    """
        STATE_UNSPECIFIED = 0
        CREATING = 1
        CONNECTED = 2
        DISCONNECTED = 3
        DELETING = 4

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. Resource labels to represent user-provided metadata.

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
    authorizedNetwork = _messages.StringField(1)
    createTime = _messages.StringField(2)
    domainResource = _messages.StringField(3)
    labels = _messages.MessageField('LabelsValue', 4)
    name = _messages.StringField(5)
    state = _messages.EnumField('StateValueValuesEnum', 6)
    statusMessage = _messages.StringField(7)
    updateTime = _messages.StringField(8)