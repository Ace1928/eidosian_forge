from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VolumeRestore(_messages.Message):
    """Represents the operation of restoring a volume from a VolumeBackup.

  Enums:
    StateValueValuesEnum: Output only. The current state of this
      VolumeRestore.
    VolumeTypeValueValuesEnum: Output only. The type of volume provisioned

  Fields:
    completeTime: Output only. The timestamp when the associated underlying
      volume restoration completed.
    createTime: Output only. The timestamp when this VolumeRestore resource
      was created.
    etag: Output only. `etag` is used for optimistic concurrency control as a
      way to help prevent simultaneous updates of a volume restore from
      overwriting each other. It is strongly suggested that systems make use
      of the `etag` in the read-modify-write cycle to perform volume restore
      updates in order to avoid race conditions.
    name: Output only. Full name of the VolumeRestore resource. Format:
      `projects/*/locations/*/restorePlans/*/restores/*/volumeRestores/*`
    state: Output only. The current state of this VolumeRestore.
    stateMessage: Output only. A human readable message explaining why the
      VolumeRestore is in its current state.
    targetPvc: Output only. The reference to the target Kubernetes PVC to be
      restored.
    uid: Output only. Server generated global unique identifier of
      [UUID](https://en.wikipedia.org/wiki/Universally_unique_identifier)
      format.
    updateTime: Output only. The timestamp when this VolumeRestore resource
      was last updated.
    volumeBackup: Output only. The full name of the VolumeBackup from which
      the volume will be restored. Format:
      `projects/*/locations/*/backupPlans/*/backups/*/volumeBackups/*`.
    volumeHandle: Output only. A storage system-specific opaque handler to the
      underlying volume created for the target PVC from the volume backup.
    volumeType: Output only. The type of volume provisioned
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The current state of this VolumeRestore.

    Values:
      STATE_UNSPECIFIED: This is an illegal state and should not be
        encountered.
      CREATING: A volume for the restore was identified and restore process is
        about to start.
      RESTORING: The volume is currently being restored.
      SUCCEEDED: The volume has been successfully restored.
      FAILED: The volume restoration process failed.
      DELETING: This VolumeRestore resource is in the process of being
        deleted.
    """
        STATE_UNSPECIFIED = 0
        CREATING = 1
        RESTORING = 2
        SUCCEEDED = 3
        FAILED = 4
        DELETING = 5

    class VolumeTypeValueValuesEnum(_messages.Enum):
        """Output only. The type of volume provisioned

    Values:
      VOLUME_TYPE_UNSPECIFIED: Default
      GCE_PERSISTENT_DISK: Compute Engine Persistent Disk volume
    """
        VOLUME_TYPE_UNSPECIFIED = 0
        GCE_PERSISTENT_DISK = 1
    completeTime = _messages.StringField(1)
    createTime = _messages.StringField(2)
    etag = _messages.StringField(3)
    name = _messages.StringField(4)
    state = _messages.EnumField('StateValueValuesEnum', 5)
    stateMessage = _messages.StringField(6)
    targetPvc = _messages.MessageField('NamespacedName', 7)
    uid = _messages.StringField(8)
    updateTime = _messages.StringField(9)
    volumeBackup = _messages.StringField(10)
    volumeHandle = _messages.StringField(11)
    volumeType = _messages.EnumField('VolumeTypeValueValuesEnum', 12)