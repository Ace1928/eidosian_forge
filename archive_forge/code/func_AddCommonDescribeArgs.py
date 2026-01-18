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
def AddCommonDescribeArgs(parser):
    """Adds common args to describe operations parsers shared across all stages.

  The common arguments are Database, Backup and OperationId.

  Args:
    parser: argparse.ArgumentParser to register arguments with.
  """
    Database(positional=False, required=False, text='For a database operation, the name of the database the operation is executing on.').AddToParser(parser)
    Backup(positional=False, required=False, text='For a backup operation, the name of the backup the operation is executing on.').AddToParser(parser)
    OperationId().AddToParser(parser)