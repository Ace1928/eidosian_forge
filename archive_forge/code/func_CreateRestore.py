from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.backup_restore.poller import BackupPoller
from googlecloudsdk.api_lib.container.backup_restore.poller import RestorePoller
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import retry
def CreateRestore(restore_ref, backup, description=None, labels=None, client=None):
    """Creates a restore resource by calling Backup for GKE service and returns a LRO."""
    if client is None:
        client = GetClientInstance()
    messages = GetMessagesModule()
    req = messages.GkebackupProjectsLocationsRestorePlansRestoresCreateRequest()
    req.restoreId = restore_ref.Name()
    req.parent = restore_ref.Parent().RelativeName()
    req.restore = messages.Restore()
    req.restore.backup = backup
    if description:
        req.restore.description = description
    if labels:
        req.restore.labels = labels
    return client.projects_locations_restorePlans_restores.Create(req)