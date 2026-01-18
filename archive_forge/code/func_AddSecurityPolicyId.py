from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddSecurityPolicyId(parser, required=True, operation=None):
    """Adds the security policy ID argument to the argparse."""
    parser.add_argument('--security-policy', required=required, help='Display name of the security policy into which the rule should be {}.'.format(operation))