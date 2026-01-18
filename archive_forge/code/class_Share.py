from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Share(_messages.Message):
    """A Filestore share.

  Enums:
    StateValueValuesEnum: Output only. The share state.

  Messages:
    LabelsValue: Resource labels to represent user provided metadata.

  Fields:
    backup: Immutable. Full name of the Cloud Filestore Backup resource that
      this Share is restored from, in the format of
      projects/{project_id}/locations/{location_id}/backups/{backup_id}.
      Empty, if the Share is created from scratch and not restored from a
      backup.
    capacityGb: File share capacity in gigabytes (GB). Filestore defines 1 GB
      as 1024^3 bytes. Must be greater than 0.
    createTime: Output only. The time when the share was created.
    description: A description of the share with 2048 characters or less.
      Requests with longer descriptions will be rejected.
    labels: Resource labels to represent user provided metadata.
    mountName: The mount name of the share. Must be 63 characters or less and
      consist of uppercase or lowercase letters, numbers, and underscores.
    name: Output only. The resource name of the share, in the format `projects
      /{project_id}/locations/{location_id}/instances/{instance_id}/shares/{sh
      are_id}`.
    nfsExportOptions: Nfs Export Options. There is a limit of 10 export
      options per file share.
    state: Output only. The share state.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The share state.

    Values:
      STATE_UNSPECIFIED: State not set.
      CREATING: Share is being created.
      READY: Share is ready for use.
      DELETING: Share is being deleted.
    """
        STATE_UNSPECIFIED = 0
        CREATING = 1
        READY = 2
        DELETING = 3

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Resource labels to represent user provided metadata.

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
    backup = _messages.StringField(1)
    capacityGb = _messages.IntegerField(2)
    createTime = _messages.StringField(3)
    description = _messages.StringField(4)
    labels = _messages.MessageField('LabelsValue', 5)
    mountName = _messages.StringField(6)
    name = _messages.StringField(7)
    nfsExportOptions = _messages.MessageField('NfsExportOptions', 8, repeated=True)
    state = _messages.EnumField('StateValueValuesEnum', 9)