from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.container.bare_metal import cluster_flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddNodeAccessConfig(parser: parser_arguments.ArgumentInterceptor):
    """Adds a command group to set the node access config.

  Args:
    parser: The argparse parser to add the flag to.
  """
    bare_metal_node_access_config_group = parser.add_group(help='Anthos on bare metal node access related settings for the standalone cluster.')
    bare_metal_node_access_config_group.add_argument('--login-user', type=str, help='User name used to access node machines.')