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
def AddBakExportStripeCountArgument(parser):
    """Add the 'stripe_count' argument to the parser for striped export."""
    parser.add_argument('--stripe_count', type=int, default=None, help='Specifies the number of stripes to use for SQL Server exports.')