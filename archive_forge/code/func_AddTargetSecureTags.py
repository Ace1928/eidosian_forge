from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddTargetSecureTags(parser, required=False):
    """Adds a target secure tag to this rule."""
    parser.add_argument('--target-secure-tags', type=arg_parsers.ArgList(), metavar='TARGET_SECURE_TAGS', required=required, help='An optional, list of target secure tags with a name of the format tagValues/ or full namespaced name')