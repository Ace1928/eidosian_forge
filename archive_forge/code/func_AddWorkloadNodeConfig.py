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
def AddWorkloadNodeConfig(parser: parser_arguments.ArgumentInterceptor):
    """Adds a command group to set the workload node config.

  Args:
    parser: The argparse parser to add the flag to.
  """
    bare_metal_workload_node_config_group = parser.add_group(help='Anthos on bare metal cluster workload node configuration.')
    _AddMaxPodsPerNode(bare_metal_workload_node_config_group)
    _AddContainerRuntime(bare_metal_workload_node_config_group)