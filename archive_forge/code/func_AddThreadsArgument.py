from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib.compute import utils as compute_utils
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util import completers
def AddThreadsArgument(parser, operation):
    """Add the 'threads' argument to the parser."""
    parser.add_argument('--threads', type=arg_parsers.BoundedInt(unlimited=True), help='Specifies the number of threads to use for the parallel {operation}. If `--parallel` is specified and this flag is not provided, Cloud SQL uses a default thread count to optimize performance.'.format(operation=operation))