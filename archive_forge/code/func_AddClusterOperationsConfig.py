from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.container.bare_metal import cluster_flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddClusterOperationsConfig(parser: parser_arguments.ArgumentInterceptor):
    """Adds a command group to set the cluster operations config.

  Args:
    parser: The argparse parser to add the flag to.
  """
    bare_metal_cluster_operations_config_group = parser.add_group(help='Anthos on bare metal standalone cluster operations configuration.')
    bare_metal_cluster_operations_config_group.add_argument('--enable-application-logs', action='store_true', help='Whether collection of application logs/metrics should be enabled (in addition to system logs/metrics).')