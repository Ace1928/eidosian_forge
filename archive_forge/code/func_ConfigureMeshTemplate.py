from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import image_utils
from googlecloudsdk.api_lib.compute import instance_template_utils
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.compute import partner_metadata_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import resource_manager_tags_utils
from googlecloudsdk.command_lib.compute.instance_templates import flags as instance_templates_flags
from googlecloudsdk.command_lib.compute.instance_templates import mesh_util
from googlecloudsdk.command_lib.compute.instance_templates import service_proxy_aux_data
from googlecloudsdk.command_lib.compute.instances import flags as instances_flags
from googlecloudsdk.command_lib.compute.resource_policies import flags as maintenance_flags
from googlecloudsdk.command_lib.compute.sole_tenancy import flags as sole_tenancy_flags
from googlecloudsdk.command_lib.compute.sole_tenancy import util as sole_tenancy_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
import six
def ConfigureMeshTemplate(args, instance_template_ref, network_interfaces):
    """Adds Anthos Service Mesh configuration into the instance template.

  Args:
      args: argparse.Namespace, An object that contains the values for the
        arguments specified in the .Args() method.
      instance_template_ref: Reference to the current instance template to be
        created.
      network_interfaces: network interfaces configured for the instance
        template.
  """
    if getattr(args, 'mesh', False):
        if args.scopes is None:
            args.scopes = constants.DEFAULT_SCOPES[:]
        if 'cloud-platform' not in args.scopes and 'https://www.googleapis.com/auth/cloud-platform' not in args.scopes:
            args.scopes.append('cloud-platform')
        workload_namespace, workload_name = mesh_util.ParseWorkload(args.mesh['workload'])
        with mesh_util.KubernetesClient(gke_cluster=args.mesh['gke-cluster']) as kube_client:
            log.status.Print('Verifying GKE cluster and Anthos Service Mesh installation...')
            namespaces = ['default', 'istio-system', workload_namespace]
            if kube_client.NamespacesExist(*namespaces) and kube_client.HasNamespaceReaderPermissions(*namespaces):
                membership_manifest = kube_client.GetMembershipCR()
                _ = kube_client.GetIdentityProviderCR()
                namespace_manifest = kube_client.GetNamespace(workload_namespace)
                workload_manifest = kube_client.GetWorkloadGroupCR(workload_namespace, workload_name)
                mesh_util.VerifyWorkloadSetup(workload_manifest)
                asm_revision = mesh_util.RetrieveWorkloadRevision(namespace_manifest)
                mesh_config = kube_client.RetrieveMeshConfig(asm_revision)
                log.status.Print('Configuring the instance template for Anthos Service Mesh...')
                project_id = instance_template_ref.project
                mesh_util.ConfigureInstanceTemplate(args, kube_client, project_id, network_interfaces[0].network, workload_namespace, workload_name, workload_manifest, membership_manifest, asm_revision, mesh_config)