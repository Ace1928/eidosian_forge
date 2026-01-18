from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.dataflow import dataflow_util
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
import six
def ArgsForSnapshotTtl(parser):
    """Register flags for specifying a snapshot ttl.

  Args:
    parser: the argparse.ArgParser to configure with a ttl argument.
  """
    parser.add_argument('--snapshot-ttl', default='7d', metavar='DURATION', type=arg_parsers.Duration(lower_bound='1h', upper_bound='30d'), help='Time to live for the snapshot.')