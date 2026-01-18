from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClusterGroup(_messages.Message):
    """A cluster group resource. `ClusterGroup` is a regional resource.

  Enums:
    StateValueValuesEnum: State of the resource.

  Messages:
    LabelsValue: You can use labels to attach lightweight metadata to
      resources for filtering and querying resource data. You can associate at
      most 64 user labels with each resource. Label keys and values may
      contain at most 63 characters and may only contain lowercase letters,
      numeric characters, underscores, and dashes. Label keys must start with
      a letter, and international characters are allowed. The empty string is
      a valid label. Labels are set on creation and updated like any other
      field. To add a new label, you must provide all of the existing labels
      along with the new label. **If you only provide a map with the new
      label, all of the old labels are removed.**

  Fields:
    createTime: Output only. Creation time of the resource.
    description: The description of this resource.
    labels: You can use labels to attach lightweight metadata to resources for
      filtering and querying resource data. You can associate at most 64 user
      labels with each resource. Label keys and values may contain at most 63
      characters and may only contain lowercase letters, numeric characters,
      underscores, and dashes. Label keys must start with a letter, and
      international characters are allowed. The empty string is a valid label.
      Labels are set on creation and updated like any other field. To add a
      new label, you must provide all of the existing labels along with the
      new label. **If you only provide a map with the new label, all of the
      old labels are removed.**
    name: Output only. The resource name of this `ClusterGroup`. Resource
      names are schemeless URIs that follow the conventions in
      https://cloud.google.com/apis/design/resource_names. For example,
      projects/ PROJECT-NUMBER/locations/us-central1/clusterGroups/MY-GROUP
    networkConfig: `NetworkConfig` passed in the param.
    nsx: Output only. NSX information.
    state: State of the resource.
    status: Output only. Deprecated. Use state instead. Status of the
      resource.
    updateTime: Output only. Update time of the resource.
    vcenter: Output only. vCenter information.
  """

    class StateValueValuesEnum(_messages.Enum):
        """State of the resource.

    Values:
      STATE_UNSPECIFIED: The default value. This value should never be used.
      ACTIVE: The cluster group is ready.
      CREATING: The cluster group is being created.
      DELETING: The cluster group is being deleted.
      UPDATING: The cluster group is being updated.
      FAILED: The cluster group has experienced an issue and might be
        unusable.
    """
        STATE_UNSPECIFIED = 0
        ACTIVE = 1
        CREATING = 2
        DELETING = 3
        UPDATING = 4
        FAILED = 5

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """You can use labels to attach lightweight metadata to resources for
    filtering and querying resource data. You can associate at most 64 user
    labels with each resource. Label keys and values may contain at most 63
    characters and may only contain lowercase letters, numeric characters,
    underscores, and dashes. Label keys must start with a letter, and
    international characters are allowed. The empty string is a valid label.
    Labels are set on creation and updated like any other field. To add a new
    label, you must provide all of the existing labels along with the new
    label. **If you only provide a map with the new label, all of the old
    labels are removed.**

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
    networkConfig = _messages.MessageField('NetworkConfig', 5)
    nsx = _messages.MessageField('Nsx', 6)
    state = _messages.EnumField('StateValueValuesEnum', 7)
    status = _messages.StringField(8)
    updateTime = _messages.StringField(9)
    vcenter = _messages.MessageField('Vcenter', 10)