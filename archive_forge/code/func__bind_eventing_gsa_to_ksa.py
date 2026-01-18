from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import collections
from googlecloudsdk.api_lib.events import iam_util
from googlecloudsdk.api_lib.kuberun.core import events_constants
from googlecloudsdk.command_lib.events import exceptions
from googlecloudsdk.command_lib.iam import iam_util as core_iam_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
def _bind_eventing_gsa_to_ksa(sa_config, client, sa_email):
    """Binds Google service account to the target eventing KSA.

  Adds the IAM policy binding roles/iam.workloadIdentityUser

  Args:
    sa_config: A ServiceAccountConfig holding the desired target kubernetes
      service account.
    client: An events/kuberun apitools.client.
    sa_email: A string of the Google service account to be bound.
  Returns: None
  """
    log.status.Print('Binding service account for {}.'.format(sa_config.description))
    control_plane_namespace = events_constants.KUBERUN_EVENTS_NAMESPACE
    project = properties.VALUES.core.project.Get(required=True)
    member = 'serviceAccount:{}.svc.id.goog[{}/{}]'.format(project, control_plane_namespace, sa_config.k8s_service_account)
    iam_util.AddIamPolicyBindingServiceAccount(sa_email, _WI_BIND_ROLE, member)
    k8s_service_account_ref = resources.REGISTRY.Parse(None, params={'namespacesId': control_plane_namespace, 'serviceaccountsId': sa_config.k8s_service_account}, collection='anthosevents.api.v1.namespaces.serviceaccounts', api_version='v1')
    client.AnnotateServiceAccount(k8s_service_account_ref, 'iam.gke.io/gcp-service-account', sa_email)
    log.status.Print('Bound service account {} to {} with {}.\n'.format(sa_email, member, _WI_BIND_ROLE))