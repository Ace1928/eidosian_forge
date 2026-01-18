from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudNetworkconnectivityV1betaGroup(_messages.Message):
    """A group represents a subset of spokes attached to a hub.

  Enums:
    StateValueValuesEnum: Output only. The current lifecycle state of this
      group.

  Messages:
    LabelsValue: Optional. Labels in key-value pair format. For more
      information about labels, see [Requirements for
      labels](https://cloud.google.com/resource-manager/docs/creating-
      managing-labels#requirements).

  Fields:
    createTime: Output only. The time the group was created.
    description: Optional. The description of the group.
    labels: Optional. Labels in key-value pair format. For more information
      about labels, see [Requirements for
      labels](https://cloud.google.com/resource-manager/docs/creating-
      managing-labels#requirements).
    name: Immutable. The name of the group. Group names must be unique. They
      use the following form: `projects/{project_number}/locations/global/hubs
      /{hub}/groups/{group_id}`
    state: Output only. The current lifecycle state of this group.
    uid: Output only. The Google-generated UUID for the group. This value is
      unique across all group resources. If a group is deleted and another
      with the same name is created, the new route table is assigned a
      different unique_id.
    updateTime: Output only. The time the group was last updated.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The current lifecycle state of this group.

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
        """Optional. Labels in key-value pair format. For more information about
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
    state = _messages.EnumField('StateValueValuesEnum', 5)
    uid = _messages.StringField(6)
    updateTime = _messages.StringField(7)