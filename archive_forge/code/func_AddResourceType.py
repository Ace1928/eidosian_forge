from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.backupdr import util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddResourceType(parser, required=True):
    """Adds a positional resource-type argument to parser.

  Args:
    parser: argparse.Parser: Parser object for command line inputs.
    required: Whether or not --resource-type is required.
  """
    parser.add_argument('--resource-type', required=required, type=str, help='Type of resource to which the backup plan should be applied.\n          E.g., `compute.<UNIVERSE_DOMAIN>.com/Instance` ')