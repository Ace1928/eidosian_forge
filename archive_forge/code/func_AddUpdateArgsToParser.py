from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddUpdateArgsToParser(parser):
    """Add flags for updating a node group to the argument parser."""
    update_node_count_group = parser.add_group(mutex=True)
    update_node_count_group.add_argument('--add-nodes', type=int, help='The number of nodes to add to the node group.')
    update_node_count_group.add_argument('--delete-nodes', metavar='NODE', type=arg_parsers.ArgList(), help='The names of the nodes to remove from the group.')
    AddNoteTemplateFlagToParser(parser, required=False)