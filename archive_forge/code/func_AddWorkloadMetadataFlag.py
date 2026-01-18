from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.compute import constants as compute_constants
from googlecloudsdk.api_lib.container import api_adapter
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.container import constants
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from Google Kubernetes Engine labels that are used for the purpose of tracking
from the node pool, depending on whether locations are being added or removed.
def AddWorkloadMetadataFlag(parser, use_mode=True):
    """Adds the --workload-metadata flag to the parser.

  Args:
    parser: A given parser.
    use_mode: Whether use Mode or NodeMetadata in WorkloadMetadataConfig.
  """
    choices = {'GCE_METADATA': "Pods running in this node pool have access to the node's underlying Compute Engine Metadata Server.", 'GKE_METADATA': 'Run the Kubernetes Engine Metadata Server on this node. The Kubernetes Engine Metadata Server exposes a metadata API to workloads that is compatible with the V1 Compute Metadata APIs exposed by the Compute Engine and App Engine Metadata Servers. This feature can only be enabled if Workload Identity is enabled at the cluster level.'}
    if not use_mode:
        choices.update({'SECURE': '[DEPRECATED] Prevents pods not in hostNetwork from accessing certain VM metadata, specifically kube-env, which contains Kubelet credentials, and the instance identity token. This is a temporary security solution available while the bootstrapping process for cluster nodes is being redesigned with significant security improvements. This feature is scheduled to be deprecated in the future and later removed.', 'EXPOSED': "[DEPRECATED] Pods running in this node pool have access to the node's underlying Compute Engine Metadata Server.", 'GKE_METADATA_SERVER': '[DEPRECATED] Run the Kubernetes Engine Metadata Server on this node. The Kubernetes Engine Metadata Server exposes a metadata API to workloads that is compatible with the V1 Compute Metadata APIs exposed by the Compute Engine and App Engine Metadata Servers. This feature can only be enabled if Workload Identity is enabled at the cluster level.'})
    parser.add_argument('--workload-metadata', default=None, choices=choices, type=lambda x: x.upper(), help='Type of metadata server available to pods running in the node pool.')
    parser.add_argument('--workload-metadata-from-node', default=None, hidden=True, choices=choices, type=lambda x: x.upper(), help='Type of metadata server available to pods running in the node pool.')