from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddRequestHeadersToAdd(parser):
    """Adds request-headers-to-add argument to the argparse."""
    parser.add_argument('--request-headers-to-add', metavar='REQUEST_HEADERS_TO_ADD', type=arg_parsers.ArgDict(), help='      A comma-separated list of header names and header values to add to\n      requests that match this rule.\n      ')