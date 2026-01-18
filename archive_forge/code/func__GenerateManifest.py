from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.fleet import gkehub_api_adapter
from googlecloudsdk.api_lib.container.fleet import gkehub_api_util
from googlecloudsdk.command_lib.container.fleet import api_util
from googlecloudsdk.command_lib.container.fleet import kube_util
from googlecloudsdk.command_lib.projects import util as p_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
def _GenerateManifest(args, service_account_key_data, image_pull_secret_data, upgrade, membership_ref, release_track=None):
    """Generate the manifest for connect agent from API.

  Args:
    args: arguments of the command.
    service_account_key_data: The contents of a Google IAM service account JSON
      file.
    image_pull_secret_data: The image pull secret content to use for private
      registries.
    upgrade: if this is an upgrade operation.
    membership_ref: The membership associated with the connect agent in the
      format of `projects/[PROJECT]/locations/global/memberships/[MEMBERSHIP]`
    release_track: the release_track used in the gcloud command,
      or None if it is not available.

  Returns:
    The full manifest to deploy the connect agent resources.
  """
    api_version = gkehub_api_util.GetApiVersionForTrack(release_track)
    delimeter = '---\n'
    full_manifest = ''
    if api_version in ['v1beta1']:
        adapter = gkehub_api_adapter.NewAPIAdapter(api_version)
        connect_agent_ref = _GetConnectAgentOptions(args, upgrade, DEFAULT_NAMESPACE, image_pull_secret_data, membership_ref)
        manifest_resources = adapter.GenerateConnectAgentManifest(connect_agent_ref)
        for resource in manifest_resources:
            full_manifest = full_manifest + (getattr(resource, 'manifest') if hasattr(resource, 'manifest') else '') + delimeter
    else:
        manifest_resources = api_util.GenerateConnectAgentManifest(membership_ref, image_pull_secret_content=image_pull_secret_data, is_upgrade=upgrade, namespace=DEFAULT_NAMESPACE, proxy=args.proxy, registry=args.docker_registry, version=args.version, release_track=release_track)
        for resource in manifest_resources.manifest:
            full_manifest = full_manifest + resource.manifest + delimeter
    full_manifest = full_manifest + CREDENTIAL_SECRET_TEMPLATE.format(namespace=DEFAULT_NAMESPACE, gcp_sa_key_secret_name=GCP_SA_KEY_SECRET_NAME, gcp_sa_key=encoding.Decode(service_account_key_data, encoding='utf8'))
    return full_manifest