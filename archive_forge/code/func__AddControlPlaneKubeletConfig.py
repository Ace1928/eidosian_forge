from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.container.gkeonprem import flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
def _AddControlPlaneKubeletConfig(bare_metal_node_pool_config_group, is_update=False):
    """Adds flags to specify the kubelet configurations in the node pool.

  Args:
    bare_metal_node_pool_config_group: The parent group to add the flags to.
    is_update: bool, whether the flag is for update command or not.
  """
    bare_metal_kubelet_config_group = bare_metal_node_pool_config_group.add_group('Modifiable kubelet configurations for bare metal machines.')
    bare_metal_kubelet_config_group.add_argument('--control-plane-registry-pull-qps', type=int, help='Limit of registry pulls per second.')
    bare_metal_kubelet_config_group.add_argument('--control-plane-registry-burst', type=int, help='Maximum size of bursty pulls, temporarily allow pulls to burst to this number, while still not exceeding registry_pull_qps.')
    _AddDisableControlPlaneSerializeImagePulls(bare_metal_kubelet_config_group, is_update=is_update)