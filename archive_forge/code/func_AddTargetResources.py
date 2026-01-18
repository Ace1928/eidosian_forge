from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddTargetResources(parser, required=False):
    """Adds the target resources the rule is applied to."""
    parser.add_argument('--target-resources', type=arg_parsers.ArgList(), metavar='TARGET_RESOURCES', required=required, help='List of URLs of target resources to which the rule is applied.')