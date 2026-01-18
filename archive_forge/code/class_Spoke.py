from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Spoke(_messages.Message):
    """A Network Connectivity Center spoke represents one or more network
  connectivity resources. When you create a spoke, you associate it with a
  hub. You must also identify a value for exactly one of the following fields:
  * linked_vpn_tunnels * linked_interconnect_attachments *
  linked_router_appliance_instances * linked_vpc_network

  Enums:
    SpokeTypeValueValuesEnum: Output only. The type of resource associated
      with the spoke.
    StateValueValuesEnum: Output only. The current lifecycle state of this
      spoke.

  Messages:
    LabelsValue: Optional labels in key-value pair format. For more
      information about labels, see [Requirements for
      labels](https://cloud.google.com/resource-manager/docs/creating-
      managing-labels#requirements).

  Fields:
    createTime: Output only. The time the spoke was created.
    description: An optional description of the spoke.
    group: Optional. The name of the group that this spoke is associated with.
    hub: Immutable. The name of the hub that this spoke is attached to.
    labels: Optional labels in key-value pair format. For more information
      about labels, see [Requirements for
      labels](https://cloud.google.com/resource-manager/docs/creating-
      managing-labels#requirements).
    linkedInterconnectAttachments: VLAN attachments that are associated with
      the spoke.
    linkedRouterApplianceInstances: Router appliance instances that are
      associated with the spoke.
    linkedVpcNetwork: Optional. VPC network that is associated with the spoke.
    linkedVpnTunnels: VPN tunnels that are associated with the spoke.
    name: Immutable. The name of the spoke. Spoke names must be unique. They
      use the following form:
      `projects/{project_number}/locations/{region}/spokes/{spoke_id}`
    reasons: Output only. The reasons for current state of the spoke. Only
      present when the spoke is in the `INACTIVE` state.
    spokeType: Output only. The type of resource associated with the spoke.
    state: Output only. The current lifecycle state of this spoke.
    uniqueId: Output only. The Google-generated UUID for the spoke. This value
      is unique across all spoke resources. If a spoke is deleted and another
      with the same name is created, the new spoke is assigned a different
      `unique_id`.
    updateTime: Output only. The time the spoke was last updated.
  """

    class SpokeTypeValueValuesEnum(_messages.Enum):
        """Output only. The type of resource associated with the spoke.

    Values:
      SPOKE_TYPE_UNSPECIFIED: Unspecified spoke type.
      VPN_TUNNEL: Spokes associated with VPN tunnels.
      INTERCONNECT_ATTACHMENT: Spokes associated with VLAN attachments.
      ROUTER_APPLIANCE: Spokes associated with router appliance instances.
      VPC_NETWORK: Spokes associated with VPC networks.
    """
        SPOKE_TYPE_UNSPECIFIED = 0
        VPN_TUNNEL = 1
        INTERCONNECT_ATTACHMENT = 2
        ROUTER_APPLIANCE = 3
        VPC_NETWORK = 4

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The current lifecycle state of this spoke.

    Values:
      STATE_UNSPECIFIED: No state information available
      CREATING: The resource's create operation is in progress.
      ACTIVE: The resource is active
      DELETING: The resource's delete operation is in progress.
      ACTIVATING: The resource's activate operation is in progress.
      DEACTIVATING: The resource's deactivate operation is in progress.
      ACCEPTING: The resource's accept operation is in progress.
      REJECTING: The resource's reject operation is in progress.
      UPDATING: The resource's update operation is in progress.
      INACTIVE: The resource is inactive.
      OBSOLETE: The hub associated with this spoke resource has been deleted.
        This state applies to spoke resources only.
    """
        STATE_UNSPECIFIED = 0
        CREATING = 1
        ACTIVE = 2
        DELETING = 3
        ACTIVATING = 4
        DEACTIVATING = 5
        ACCEPTING = 6
        REJECTING = 7
        UPDATING = 8
        INACTIVE = 9
        OBSOLETE = 10

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional labels in key-value pair format. For more information about
    labels, see [Requirements for labels](https://cloud.google.com/resource-
    manager/docs/creating-managing-labels#requirements).

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
    description = _messages.StringField(2)
    group = _messages.StringField(3)
    hub = _messages.StringField(4)
    labels = _messages.MessageField('LabelsValue', 5)
    linkedInterconnectAttachments = _messages.MessageField('LinkedInterconnectAttachments', 6)
    linkedRouterApplianceInstances = _messages.MessageField('LinkedRouterApplianceInstances', 7)
    linkedVpcNetwork = _messages.MessageField('LinkedVpcNetwork', 8)
    linkedVpnTunnels = _messages.MessageField('LinkedVpnTunnels', 9)
    name = _messages.StringField(10)
    reasons = _messages.MessageField('StateReason', 11, repeated=True)
    spokeType = _messages.EnumField('SpokeTypeValueValuesEnum', 12)
    state = _messages.EnumField('StateValueValuesEnum', 13)
    uniqueId = _messages.StringField(14)
    updateTime = _messages.StringField(15)