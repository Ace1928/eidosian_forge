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
def InstancePartition(positional=True, required=True, hidden=True, text='Cloud Spanner instance partition ID.'):
    """Initialize an instance partition flag.

  Args:
    positional: bool. If true, then it's a positional flag.
    required: bool. If true, then this flag is required.
    hidden: bool. If true, then this flag is hidden.
    text: helper test.

  Returns:
  """
    if positional:
        return base.Argument('instance_partition', completer=InstancePartitionCompleter, hidden=hidden, help=text)
    else:
        return base.Argument('--instance-partition', required=required, hidden=hidden, help=text)