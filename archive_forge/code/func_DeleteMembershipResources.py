from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.container.fleet import kube_util
from googlecloudsdk.core import exceptions
def DeleteMembershipResources(kube_client):
    """Deletes the Membership CRD.

  Due to garbage collection all Membership resources will also be deleted.

  Args:
    kube_client: A KubernetesClient.
  """
    try:
        succeeded, error = waiter.WaitFor(kube_util.KubernetesPoller(), MembershipCRDeleteOperation(kube_client), 'Deleting membership CR in the cluster', pre_start_sleep_ms=kube_util.NAMESPACE_DELETION_INITIAL_WAIT_MS, max_wait_ms=kube_util.NAMESPACE_DELETION_TIMEOUT_MS, wait_ceiling_ms=kube_util.NAMESPACE_DELETION_MAX_POLL_INTERVAL_MS, sleep_ms=kube_util.NAMESPACE_DELETION_INITIAL_POLL_INTERVAL_MS)
    except waiter.TimeoutError:
        raise exceptions.Error('Timeout deleting membership CR from cluster.')
    if not succeeded:
        raise exceptions.Error('Could not delete membership CR from cluster. Error: {}'.format(error))