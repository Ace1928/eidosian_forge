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
def WaitForRestoreToFinish(restore, max_wait_ms=1800000, exponential_sleep_multiplier=1.4, jitter_ms=1000, wait_ceiling_ms=180000, status_update=_RestoreStatusUpdate, sleep_ms=2000, client=None):
    """Waits for restore resource to be terminal state."""
    if not client:
        client = GetClientInstance()
    messages = GetMessagesModule()
    retryer = retry.Retryer(max_retrials=None, max_wait_ms=max_wait_ms, exponential_sleep_multiplier=exponential_sleep_multiplier, jitter_ms=jitter_ms, wait_ceiling_ms=wait_ceiling_ms, status_update_func=status_update)
    restore_poller = RestorePoller(client, messages)
    try:
        result = retryer.RetryOnResult(func=restore_poller.Poll, args=(restore,), should_retry_if=restore_poller.IsNotDone, sleep_ms=sleep_ms)
        log.Print('Restore completed. Restore state: {0}'.format(result.state))
        return result
    except retry.WaitException:
        raise WaitForCompletionTimeoutError('Timeout waiting for restore to complete. Restore is not completed, use "gcloud container backup-restore restores describe" command to check restore status.')