from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudNetworkconnectivityV1betaHub(_messages.Message):
    """A Network Connectivity Center hub is a global management resource to
  which you attach spokes. A single hub can contain spokes from multiple
  regions. However, if any of a hub's spokes use the site-to-site data
  transfer feature, the resources associated with those spokes must all be in
  the same VPC network. Spokes that do not use site-to-site data transfer can
  be associated with any VPC network in your project.

  Enums:
    StateValueValuesEnum: Output only. The current lifecycle state of this
      hub.

  Messages:
    LabelsValue: Optional labels in key-value pair format. For more
      information about labels, see [Requirements for
      labels](https://cloud.google.com/resource-manager/docs/creating-
      managing-labels#requirements).

  Fields:
    createTime: Output only. The time the hub was created.
    description: An optional description of the hub.
    labels: Optional labels in key-value pair format. For more information
      about labels, see [Requirements for
      labels](https://cloud.google.com/resource-manager/docs/creating-
      managing-labels#requirements).
    name: Immutable. The name of the hub. Hub names must be unique. They use
      the following form:
      `projects/{project_number}/locations/global/hubs/{hub_id}`
    routeTables: Output only. The route tables that belong to this hub. They
      use the following form: `projects/{project_number}/locations/global/hubs
      /{hub_id}/routeTables/{route_table_id}` This field is read-only. Network
      Connectivity Center automatically populates it based on the route tables
      nested under the hub.
    routingVpcs: The VPC networks associated with this hub's spokes. This
      field is read-only. Network Connectivity Center automatically populates
      it based on the set of spokes attached to the hub.
    spokeSummary: Output only. A summary of the spokes associated with a hub.
      The summary includes a count of spokes according to type and according
      to state. If any spokes are inactive, the summary also lists the reasons
      they are inactive, including a count for each reason.
    state: Output only. The current lifecycle state of this hub.
    uniqueId: Output only. The Google-generated UUID for the hub. This value
      is unique across all hub resources. If a hub is deleted and another with
      the same name is created, the new hub is assigned a different unique_id.
    updateTime: Output only. The time the hub was last updated.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The current lifecycle state of this hub.

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
    labels = _messages.MessageField('LabelsValue', 3)
    name = _messages.StringField(4)
    routeTables = _messages.StringField(5, repeated=True)
    routingVpcs = _messages.MessageField('GoogleCloudNetworkconnectivityV1betaRoutingVPC', 6, repeated=True)
    spokeSummary = _messages.MessageField('GoogleCloudNetworkconnectivityV1betaSpokeSummary', 7)
    state = _messages.EnumField('StateValueValuesEnum', 8)
    uniqueId = _messages.StringField(9)
    updateTime = _messages.StringField(10)