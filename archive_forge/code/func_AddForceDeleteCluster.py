from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.container.gkeonprem import flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def AddForceDeleteCluster(parser: parser_arguments.ArgumentInterceptor):
    """Adds a flag for force delete cluster operation when there are existing node pools.

  Args:
    parser: The argparse parser to add the flag to.
  """
    parser.add_argument('--force', action='store_true', help='If set, any node pools from the cluster will also be deleted. This flag is required if the cluster has any associated node pools.')