from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClusterGroupBackup(_messages.Message):
    """Backup resource of the `ClusterGroup`.

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
    clusterGroupId: Identity of the `ClusterGroup` of which this resource is a
      backup.
    createTime: Output only. Creation time of the resource.
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
    name: Output only. The resource name of this `ClusterGroupBackup`.
      Resource names are schemeless URIs that follow the conventions in
      https://cloud.google.com/apis/design/resource_names. For example,
      projects/ PROJECT-NUMBER/locations/us-central1/clusterGroupBackups/MY-
      BACKUP
    updateTime: Output only. Update time of the resource.
  """

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
    clusterGroupId = _messages.StringField(1)
    createTime = _messages.StringField(2)
    labels = _messages.MessageField('LabelsValue', 3)
    name = _messages.StringField(4)
    updateTime = _messages.StringField(5)