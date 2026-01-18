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
def AddSqlServerTimeZone(parser, hidden=False):
    """Adds the `--time-zone` flag to the parser."""
    parser.add_argument('--time-zone', required=False, help='Set a non-default time zone. Only available for SQL Server instances.', hidden=hidden)