from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def _GetBackup(self, backup):
    req = self.messages.GkebackupProjectsLocationsBackupPlansBackupsGetRequest()
    req.name = backup
    return self.client.projects_locations_backupPlans_backups.Get(req)