from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.core import log
def AddFailover(parser, default):
    """Adds the failover argument to the argparse."""
    parser.add_argument('--failover', action='store_true', default=default, help='      Designates whether this is a failover backend. More than one\n      failover backend can be configured for a given BackendService.\n      Not compatible with the --global flag')