from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.filestore.backups import util as backup_util
from googlecloudsdk.command_lib.filestore.snapshots import util as snapshot_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
class AlphaFilestoreAdapter(object):
    """Adapter for the alpha filestore API."""

    def __init__(self):
        self.client = GetClient(version=ALPHA_API_VERSION)
        self.messages = GetMessages(version=ALPHA_API_VERSION)

    def ParseFileShareIntoInstance(self, instance, file_share, instance_zone=None):
        """Parse specified file share configs into an instance message.

    Args:
      instance: The Filestore instance object.
      file_share: File share configuration.
      instance_zone: The instance zone.

    Raises:
      InvalidArgumentError: If file_share argument constraints are violated.

    """
        if instance.fileShares is None:
            instance.fileShares = []
        if file_share:
            source_snapshot = None
            source_backup = None
            location = None
            instance.fileShares = [fs for fs in instance.fileShares if fs.name != file_share.get('name')]
            if 'source-snapshot' in file_share:
                project = properties.VALUES.core.project.Get(required=True)
                location = file_share.get('source-snapshot-region') or instance_zone
                source_snapshot = snapshot_util.SNAPSHOT_NAME_TEMPLATE.format(project, location, file_share.get('source-snapshot'))
            if 'source-backup' in file_share:
                project = properties.VALUES.core.project.Get(required=True)
                location = file_share.get('source-backup-region')
                source_backup = backup_util.BACKUP_NAME_TEMPLATE.format(project, location, file_share.get('source-backup'))
            if None not in [source_snapshot, source_backup]:
                raise InvalidArgumentError("At most one of ['source-snapshot', 'source-backup'] can be specified.")
            if source_backup is not None and location is None:
                raise InvalidArgumentError("If 'source-backup' is specified, 'source-backup-region' must also be specified.")
            nfs_export_options = FilestoreClient.MakeNFSExportOptionsMsg(self.messages, file_share.get('nfs-export-options', []))
            file_share_config = self.messages.FileShareConfig(name=file_share.get('name'), capacityGb=utils.BytesToGb(file_share.get('capacity')), sourceSnapshot=source_snapshot, sourceBackup=source_backup, nfsExportOptions=nfs_export_options)
            instance.fileShares.append(file_share_config)

    def FileSharesFromInstance(self, instance):
        """Get file share configs from instance message."""
        return instance.fileShares

    def ParseUpdatedInstanceConfig(self, instance_config, description=None, labels=None, file_share=None, managed_ad=None, disconnect_managed_ad=None, clear_nfs_export_options=False):
        """Parse update information into an updated Instance message."""
        if description:
            instance_config.description = description
        if labels:
            instance_config.labels = labels
        if file_share:
            self.ValidateFileShareForUpdate(instance_config, file_share)
            orig_nfs_export_options = []
            if instance_config.fileShares[0] and instance_config.fileShares[0].nfsExportOptions:
                orig_nfs_export_options = instance_config.fileShares[0].nfsExportOptions
            self.ParseFileShareIntoInstance(instance_config, file_share)
            if not instance_config.fileShares[0].nfsExportOptions:
                instance_config.fileShares[0].nfsExportOptions = orig_nfs_export_options
            if clear_nfs_export_options:
                instance_config.fileShares[0].nfsExportOptions = []
        if managed_ad:
            self.ParseManagedADIntoInstance(instance_config, managed_ad)
        if disconnect_managed_ad:
            instance_config.directoryServices = None
        return instance_config

    def ValidateFileShareForUpdate(self, instance_config, file_share):
        """Validate the updated file share configuration.

    The new config must have the same name as the existing config.

    Args:
      instance_config: Instance message for existing instance.
      file_share: dict with keys 'name' and 'capacity'.

    Raises:
      InvalidNameError: If the names don't match.
      ValueError: If the instance doesn't have an existing file share.
    """
        existing = self.FileSharesFromInstance(instance_config)
        if not existing:
            raise ValueError('Existing instance does not have file shares configured')
        existing_file_share = existing[0]
        if existing_file_share.name != file_share.get('name'):
            raise InvalidNameError('Must update an existing file share. Existing file share is named [{}]. Requested update had name [{}].'.format(existing_file_share.name, file_share.get('name')))

    def UpdateInstance(self, instance_ref, instance_config, update_mask):
        """Send a Patch request for the Cloud Filestore instance."""
        update_request = self.messages.FileProjectsLocationsInstancesPatchRequest(instance=instance_config, name=instance_ref.RelativeName(), updateMask=update_mask)
        update_op = self.client.projects_locations_instances.Patch(update_request)
        return update_op

    def ParseConnectMode(self, network_config, key):
        """Parse and match the supplied connection mode."""
        try:
            value = self.messages.NetworkConfig.ConnectModeValueValuesEnum.lookup_by_name(key)
        except KeyError:
            raise InvalidArgumentError('[{}] is not a valid connect-mode. Must be one of DIRECT_PEERING or PRIVATE_SERVICE_ACCESS.'.format(key))
        else:
            network_config.connectMode = value