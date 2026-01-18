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
class FilestoreAdapter(BetaFilestoreAdapter):
    """Adapter for the filestore v1 API."""

    def __init__(self):
        super(FilestoreAdapter, self).__init__()
        self.client = GetClient(version=V1_API_VERSION)
        self.messages = GetMessages(version=V1_API_VERSION)

    def ParseFileShareIntoInstance(self, instance, file_share, instance_zone=None):
        """Parse specified file share configs into an instance message."""
        del instance_zone
        if instance.fileShares is None:
            instance.fileShares = []
        if file_share:
            instance.fileShares = [fs for fs in instance.fileShares if fs.name != file_share.get('name')]
            source_backup = self._ParseSourceBackupFromFileshare(file_share)
            nfs_export_options = FilestoreClient.MakeNFSExportOptionsMsg(self.messages, file_share.get('nfs-export-options', []))
            file_share_config = self.messages.FileShareConfig(name=file_share.get('name'), capacityGb=utils.BytesToGb(file_share.get('capacity')), sourceBackup=source_backup, nfsExportOptions=nfs_export_options)
            instance.fileShares.append(file_share_config)