from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
class BackupPoller(object):
    """Backup poller for polling backup until it's terminal."""

    def __init__(self, client, messages):
        self.client = client
        self.messages = messages

    def IsNotDone(self, backup, unused_state):
        del unused_state
        return not (backup.state == self.messages.Backup.StateValueValuesEnum.SUCCEEDED or backup.state == self.messages.Backup.StateValueValuesEnum.FAILED or backup.state == self.messages.Backup.StateValueValuesEnum.DELETING)

    def _GetBackup(self, backup):
        req = self.messages.GkebackupProjectsLocationsBackupPlansBackupsGetRequest()
        req.name = backup
        return self.client.projects_locations_backupPlans_backups.Get(req)

    def Poll(self, backup):
        return self._GetBackup(backup)

    def GetResult(self, backup):
        return self._GetBackup(backup)