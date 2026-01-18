from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import datetime
from googlecloudsdk.api_lib.logging import util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def LogFilterPositionalArgs(parser):
    """Add a log filter positional arg."""
    parser.add_argument('log_filter', help=_LOG_FILTER_HELP_TEXT, nargs='?')