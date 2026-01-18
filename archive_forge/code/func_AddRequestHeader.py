from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddRequestHeader(parser, is_add):
    """Adds request-header-to-exclude argument to the argparse."""
    parser.add_argument('--request-header-to-exclude', type=arg_parsers.ArgDict(spec={'op': str, 'val': str}, required_keys=['op']), action='append', help=_WAF_EXCLUSION_REQUEST_HEADER_HELP_TEXT_FOR_ADD if is_add else _WAF_EXCLUSION_REQUEST_HEADER_HELP_TEXT_FOR_REMOVE)