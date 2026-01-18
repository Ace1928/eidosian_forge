from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AllowedClient(_messages.Message):
    """Represents an 'access point' for the share.

  Enums:
    MountPermissionsValueValuesEnum: Mount permissions.

  Fields:
    allowDev: Allow dev flag. Which controls whether to allow creation of
      devices.
    allowSuid: Allow the setuid flag.
    allowedClientsCidr: The subnet of IP addresses permitted to access the
      share.
    mountPermissions: Mount permissions.
    network: The network the access point sits on.
    nfsPath: Output only. The path to access NFS, in format
      shareIP:/InstanceID InstanceID is the generated ID instead of customer
      provided name. example like "10.0.0.0:/g123456789-nfs001"
    noRootSquash: Disable root squashing, which is a feature of NFS. Root
      squash is a special mapping of the remote superuser (root) identity when
      using identity authentication.
    shareIp: Output only. The IP address of the share on this network.
      Assigned automatically during provisioning based on the network's
      services_cidr.
  """

    class MountPermissionsValueValuesEnum(_messages.Enum):
        """Mount permissions.

    Values:
      MOUNT_PERMISSIONS_UNSPECIFIED: Permissions were not specified.
      READ: NFS share can be mount with read-only permissions.
      READ_WRITE: NFS share can be mount with read-write permissions.
    """
        MOUNT_PERMISSIONS_UNSPECIFIED = 0
        READ = 1
        READ_WRITE = 2
    allowDev = _messages.BooleanField(1)
    allowSuid = _messages.BooleanField(2)
    allowedClientsCidr = _messages.StringField(3)
    mountPermissions = _messages.EnumField('MountPermissionsValueValuesEnum', 4)
    network = _messages.StringField(5)
    nfsPath = _messages.StringField(6)
    noRootSquash = _messages.BooleanField(7)
    shareIp = _messages.StringField(8)