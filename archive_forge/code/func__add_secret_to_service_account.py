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
def _add_secret_to_service_account(client, sa_config, product_type, sa_email):
    """Adds new secret to service account.

  Args:
    client: An api_tools client.
    sa_config: A ServiceAccountConfig.
    product_type: events_constants.Product enum.
    sa_email: String of the targeted service account email.
  """
    control_plane_namespace = events_constants.ControlPlaneNamespaceFromProductType(product_type)
    secret_ref = resources.REGISTRY.Parse(sa_config.secret_name, params={'namespacesId': control_plane_namespace}, collection='run.api.v1.namespaces.secrets', api_version='v1')
    service_account_ref = resources.REGISTRY.Parse(sa_email, params={'projectsId': '-'}, collection=core_iam_util.SERVICE_ACCOUNTS_COLLECTION)
    prompt_if_can_prompt('This will create a new key for the service account [{}].'.format(sa_email))
    _, key_ref = client.CreateOrReplaceServiceAccountSecret(secret_ref, service_account_ref)
    log.status.Print('Added key [{}] to cluster for [{}].'.format(key_ref.Name(), sa_email))