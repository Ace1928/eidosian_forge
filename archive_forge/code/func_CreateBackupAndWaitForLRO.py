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
def CreateBackupAndWaitForLRO(backup_ref, description=None, labels=None, retain_days=None, delete_lock_days=None, client=None):
    """Creates a backup resource and wait for the resulting LRO to complete."""
    if client is None:
        client = GetClientInstance()
    operation = CreateBackup(backup_ref, description=description, labels=labels, retain_days=retain_days, delete_lock_days=delete_lock_days, client=client)
    operation_ref = resources.REGISTRY.ParseRelativeName(operation.name, 'gkebackup.projects.locations.operations')
    log.CreatedResource(operation_ref.RelativeName(), kind='backup {0}'.format(backup_ref.Name()), is_async=True)
    poller = waiter.CloudOperationPollerNoResources(client.projects_locations_operations)
    return waiter.WaitFor(poller, operation_ref, 'Creating backup {0}'.format(backup_ref.Name()))