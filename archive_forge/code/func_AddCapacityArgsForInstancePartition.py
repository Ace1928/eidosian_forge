from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from argcomplete.completers import FilesCompleter
from cloudsdk.google.protobuf import descriptor_pb2
from googlecloudsdk.api_lib.spanner import databases
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.spanner import ddl_parser
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.core.util import files
def AddCapacityArgsForInstancePartition(parser):
    """Parse the instance partition capacity arguments.

  Args:
    parser: the argparse parser for the command.
  """
    capacity_parser = parser.add_argument_group(mutex=True, required=False)
    Nodes(text='Number of nodes for the instance partition.').AddToParser(capacity_parser)
    ProcessingUnits(text='Number of processing units for the instance partition.').AddToParser(capacity_parser)