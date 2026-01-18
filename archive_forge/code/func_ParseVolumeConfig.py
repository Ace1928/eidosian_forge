from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.netapp import constants
from googlecloudsdk.api_lib.netapp import util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def ParseVolumeConfig(self, name=None, capacity=None, description=None, storage_pool=None, protocols=None, share_name=None, export_policy=None, unix_permissions=None, smb_settings=None, snapshot_policy=None, snap_reserve=None, snapshot_directory=None, security_style=None, enable_kerberos=None, snapshot=None, backup=None, restricted_actions=None, backup_config=None, large_capacity=None, multiple_endpoints=None, tiering_policy=None, labels=None):
    """Parses the command line arguments for Create Volume into a config.

    Args:
      name: the name of the Volume
      capacity: the storage capacity of the Volume.
      description: the description of the Volume.
      storage_pool: the Storage Pool the Volume is attached to.
      protocols: the type of fileshare protocol of the Volume.
      share_name: the share name or mount point of the Volume.
      export_policy: the export policy of the Volume if NFS.
      unix_permissions: the Unix permissions for the Volume.
      smb_settings: the SMB settings for the Volume.
      snapshot_policy: the Snapshot Policy for the Volume
      snap_reserve: the snap reserve (double) for the Volume
      snapshot_directory: Bool on whether to use snapshot directory for Volume
      security_style: the security style of the Volume
      enable_kerberos: Bool on whether to use kerberos for Volume
      snapshot: the snapshot name to create Volume from
      backup: the backup to create the Volume from.
      restricted_actions: the actions to be restricted on a Volume
      backup_config: the Backup Config attached to the Volume
      large_capacity: Bool on whether to use large capacity for Volume
      multiple_endpoints: Bool on whether to use multiple endpoints for Volume
      tiering_policy: the tiering policy for the volume.
      labels: the parsed labels value.

    Returns:
      the configuration that will be used as the request body for creating a
      Cloud NetApp Files Volume.
    """
    volume = self.messages.Volume()
    volume.name = name
    volume.capacityGib = capacity
    volume.description = description
    volume.labels = labels
    volume.storagePool = storage_pool
    volume.shareName = share_name
    self.ParseExportPolicy(volume, export_policy)
    self.ParseProtocols(volume, protocols)
    volume.unixPermissions = unix_permissions
    volume.smbSettings = smb_settings
    self.ParseSnapshotPolicy(volume, snapshot_policy)
    volume.snapReserve = snap_reserve
    volume.snapshotDirectory = snapshot_directory
    volume.securityStyle = security_style
    volume.kerberosEnabled = enable_kerberos
    restore_parameters = self.messages.RestoreParameters()
    if snapshot is not None:
        restore_parameters.sourceSnapshot = snapshot
    if backup is not None:
        restore_parameters.sourceBackup = backup
    if backup is None and snapshot is None:
        restore_parameters = None
    volume.restoreParameters = restore_parameters
    volume.restrictedActions = restricted_actions
    if backup_config is not None:
        self.ParseBackupConfig(volume, backup_config)
    if large_capacity is not None:
        volume.largeCapacity = large_capacity
    if multiple_endpoints is not None:
        volume.multipleEndpoints = multiple_endpoints
    if tiering_policy is not None:
        self.ParseTieringPolicy(volume, tiering_policy)
    return volume