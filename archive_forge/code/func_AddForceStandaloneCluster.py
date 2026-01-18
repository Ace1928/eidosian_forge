from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.container.bare_metal import cluster_flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddForceStandaloneCluster(parser: parser_arguments.ArgumentInterceptor):
    """Adds a flag for force standalone cluster operation when there are existing node pools.

  Args:
    parser: The argparse parser to add the flag to.
  """
    parser.add_argument('--force', action='store_true', help='If set, the operation will also apply to the child node pools. This flag is required if the cluster has any associated node pools.')