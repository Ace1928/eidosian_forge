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
def DeployConnectAgent(kube_client, args, service_account_key_data, image_pull_secret_data, membership_ref, release_track=None):
    """Deploys the Connect Agent to the cluster.

  Args:
    kube_client: A Kubernetes Client for the cluster to be registered.
    args: arguments of the command.
    service_account_key_data: The contents of a Google IAM service account JSON
      file
    image_pull_secret_data: The contents of image pull secret to use for
      private registries.
    membership_ref: The membership should be associated with the connect agent
      in the format of
      `project/[PROJECT]/location/global/memberships/[MEMBERSHIP]`.
    release_track: the release_track used in the gcloud command,
      or None if it is not available.
  Raises:
    exceptions.Error: If the agent cannot be deployed properly
    calliope_exceptions.MinimumArgumentException: If the agent cannot be
    deployed properly
  """
    project_id = properties.VALUES.core.project.GetOrFail()
    log.status.Print('Generating the Connect Agent manifest...')
    full_manifest = _GenerateManifest(args, service_account_key_data, image_pull_secret_data, False, membership_ref, release_track)
    if args.manifest_output_file:
        try:
            files.WriteFileContents(files.ExpandHomeDir(args.manifest_output_file), full_manifest, private=True)
        except files.Error as e:
            raise exceptions.Error('Could not create manifest file: {}'.format(e))
        log.status.Print(MANIFEST_SAVED_MESSAGE.format(args.manifest_output_file))
        return
    namespaces = _GKEConnectNamespace(kube_client, project_id)
    if len(namespaces) > 1:
        raise exceptions.Error('Multiple namespaces [{}] containing the Connect Agent found incluster [{}]. Cannot deploy a new Connect Agent'.format(namespaces, args.MEMBERSHIP_NAME))
    namespace = namespaces[0]
    log.status.Print('Deploying the Connect Agent on cluster [{}] in namespace [{}]...'.format(args.MEMBERSHIP_NAME, namespace))
    kube_util.DeleteNamespace(kube_client, namespace)
    _PurgeAlphaInstaller(kube_client, namespace, project_id)
    _, err = kube_client.Apply(full_manifest)
    if err:
        raise exceptions.Error('Failed to apply manifest to cluster: {}'.format(err))
    log.status.Print('Deployed the Connect Agent on cluster [{}] in namespace [{}].'.format(args.MEMBERSHIP_NAME, namespace))