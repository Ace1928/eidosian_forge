from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudNetworkconnectivityV1betaRoute(_messages.Message):
    """A route defines a path from VM instances within a spoke to a specific
  destination resource. Only VPC spokes have routes.

  Enums:
    StateValueValuesEnum: Output only. The current lifecycle state of the
      route.
    TypeValueValuesEnum: Output only. The route's type. Its type is determined
      by the properties of its IP address range.

  Messages:
    LabelsValue: Optional labels in key-value pair format. For more
      information about labels, see [Requirements for
      labels](https://cloud.google.com/resource-manager/docs/creating-
      managing-labels#requirements).

  Fields:
    createTime: Output only. The time the route was created.
    description: An optional description of the route.
    ipCidrRange: The destination IP address range.
    labels: Optional labels in key-value pair format. For more information
      about labels, see [Requirements for
      labels](https://cloud.google.com/resource-manager/docs/creating-
      managing-labels#requirements).
    location: Output only. The origin location of the route. Uses the
      following form: "projects/{project}/locations/{location}" Example:
      projects/1234/locations/us-central1
    name: Immutable. The name of the route. Route names must be unique. Route
      names use the following form: `projects/{project_number}/locations/globa
      l/hubs/{hub}/routeTables/{route_table_id}/routes/{route_id}`
    nextHopVpcNetwork: Immutable. The destination VPC network for packets on
      this route.
    spoke: Immutable. The spoke that this route leads to. Example:
      projects/12345/locations/global/spokes/SPOKE
    state: Output only. The current lifecycle state of the route.
    type: Output only. The route's type. Its type is determined by the
      properties of its IP address range.
    uid: Output only. The Google-generated UUID for the route. This value is
      unique across all Network Connectivity Center route resources. If a
      route is deleted and another with the same name is created, the new
      route is assigned a different `uid`.
    updateTime: Output only. The time the route was last updated.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The current lifecycle state of the route.

    Values:
      STATE_UNSPECIFIED: No state information available
      CREATING: The resource's create operation is in progress.
      ACTIVE: The resource is active
      DELETING: The resource's delete operation is in progress.
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
        ACCEPTING = 4
        REJECTING = 5
        UPDATING = 6
        INACTIVE = 7
        OBSOLETE = 8

    class TypeValueValuesEnum(_messages.Enum):
        """Output only. The route's type. Its type is determined by the
    properties of its IP address range.

    Values:
      ROUTE_TYPE_UNSPECIFIED: No route type information specified
      VPC_PRIMARY_SUBNET: The route leads to a destination within the primary
        address range of the VPC network's subnet.
      VPC_SECONDARY_SUBNET: The route leads to a destination within the
        secondary address range of the VPC network's subnet.
    """
        ROUTE_TYPE_UNSPECIFIED = 0
        VPC_PRIMARY_SUBNET = 1
        VPC_SECONDARY_SUBNET = 2

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
    ipCidrRange = _messages.StringField(3)
    labels = _messages.MessageField('LabelsValue', 4)
    location = _messages.StringField(5)
    name = _messages.StringField(6)
    nextHopVpcNetwork = _messages.MessageField('GoogleCloudNetworkconnectivityV1betaNextHopVpcNetwork', 7)
    spoke = _messages.StringField(8)
    state = _messages.EnumField('StateValueValuesEnum', 9)
    type = _messages.EnumField('TypeValueValuesEnum', 10)
    uid = _messages.StringField(11)
    updateTime = _messages.StringField(12)