from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddTargetServiceAccounts(parser, required=False):
    """Adds the target service accounts for the rule."""
    parser.add_argument('--target-service-accounts', type=arg_parsers.ArgList(), metavar='TARGET_SERVICE_ACCOUNTS', required=required, help='List of target service accounts for the rule.')