from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def _GetRestore(self, restore):
    req = self.messages.GkebackupProjectsLocationsRestorePlansRestoresGetRequest()
    req.name = restore
    return self.client.projects_locations_restorePlans_restores.Get(req)