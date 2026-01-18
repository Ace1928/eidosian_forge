from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddArgsListAssociation(parser):
    """Adds the arguments of association list."""
    group = parser.add_group(required=True, mutex=True)
    group.add_argument('--organization', help='ID of the organization with which the association is listed.')
    group.add_argument('--folder', help='ID of the folder with which the association is listed.')