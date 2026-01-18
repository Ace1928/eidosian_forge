from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.netapp import constants
from googlecloudsdk.api_lib.netapp import util as netapp_util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
class BackupsAdapter(object):
    """Adapter for the GA Cloud NetApp Files API for Backups."""

    def __init__(self):
        self.release_track = base.ReleaseTrack.GA
        self.client = netapp_util.GetClientInstance(release_track=self.release_track)
        self.messages = netapp_util.GetMessagesModule(release_track=self.release_track)

    def ParseUpdatedBackup(self, backup, description=None, labels=None):
        """Parses updates into a new Backup."""
        if description is not None:
            backup.description = description
        if labels is not None:
            backup.labels = labels
        return backup

    def UpdateBackup(self, backup_ref, backup, update_mask):
        """Send a Patch request for the Cloud NetApp Backup."""
        update_request = self.messages.NetappProjectsLocationsBackupVaultsBackupsPatchRequest(backup=backup, name=backup_ref.RelativeName(), updateMask=update_mask)
        update_op = self.client.projects_locations_backupVaults_backups.Patch(update_request)
        return update_op