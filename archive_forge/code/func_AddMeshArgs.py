from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import ipaddress
import re
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute.instance_templates import service_proxy_aux_data
def AddMeshArgs(parser, hide_arguments=False):
    """Adds Anthos Service Mesh configuration arguments for instance templates."""
    mesh_group = parser.add_group(hidden=hide_arguments)
    mesh_group.add_argument('--mesh', type=arg_parsers.ArgDict(spec={'gke-cluster': str, 'workload': str}, allow_key_only=False, required_keys=['gke-cluster', 'workload']), hidden=hide_arguments, help='      Controls whether the Anthos Service Mesh service proxy (Envoy) and agent are installed and configured on the VM.\n      "cloud-platform" scope is enabled automatically to allow the service proxy to be started.\n      Do not use the `--no-scopes` flag.\n\n      *gke-cluster*::: The location/name of the GKE cluster. The location can be a zone or a\n          region, e.g. ``us-central1-a/my-cluster\'\'.\n\n      *workload*::: The workload identifier of the VM. In a GKE cluster, it is\n          the identifier namespace/name of the `WorkloadGroup` custom resource representing the VM\n          workload, e.g. ``foo/my-workload\'\'.\n      ')