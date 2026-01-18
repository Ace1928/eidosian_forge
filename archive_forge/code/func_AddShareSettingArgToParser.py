from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddShareSettingArgToParser(parser):
    """Add share setting configuration arguments to parser."""
    group = parser.add_group(help='Manage the properties of a shared setting')
    group.add_argument('--share-setting', required=True, choices=['projects', 'organization', 'local'], help='\nSpecify if this node group is shared; and if so, the type of sharing:\nshare with specific projects or folders.\n')
    group.add_argument('--share-with', type=arg_parsers.ArgList(min_length=1), metavar='PROJECT', help='A list of specific projects this node group should be shared with.')