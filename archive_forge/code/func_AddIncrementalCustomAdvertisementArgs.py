from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddIncrementalCustomAdvertisementArgs(parser, resource_str):
    """Adds common arguments for incrementally updating custom advertisements."""
    incremental_args = parser.add_mutually_exclusive_group(required=False)
    incremental_args.add_argument('--add-advertisement-groups', type=arg_parsers.ArgList(choices=_GROUP_CHOICES, element_type=lambda group: group.upper()), metavar='GROUP', help='A list of pre-defined groups of IP ranges to dynamically advertise\n              on this {0}. This list is appended to any existing advertisements.\n              This field can only be specified in custom advertisement mode.'.format(resource_str))
    incremental_args.add_argument('--remove-advertisement-groups', type=arg_parsers.ArgList(choices=_GROUP_CHOICES, element_type=lambda group: group.upper()), metavar='GROUP', help='A list of pre-defined groups of IP ranges to remove from dynamic\n              advertisement on this {0}. Each group in the list must exist in\n              the current set of custom advertisements. This field can only be\n              specified in custom advertisement mode.'.format(resource_str))
    incremental_args.add_argument('--add-advertisement-ranges', type=arg_parsers.ArgDict(allow_key_only=True), metavar='CIDR_RANGE=DESC', help='A list of individual IP ranges, in CIDR format, to dynamically\n              advertise on this {0}. This list is appended to any existing\n              advertisements. Each IP range can (optionally) be given a text\n              description DESC. For example, to advertise a specific range, use\n              `--advertisement-ranges=192.168.10.0/24`.  To store a description\n              with the range, use\n              `--advertisement-ranges=192.168.10.0/24=my-networks`. This list\n              can only be specified in custom advertisement mode.'.format(resource_str))
    incremental_args.add_argument('--remove-advertisement-ranges', type=arg_parsers.ArgList(), metavar='CIDR_RANGE', help='A list of individual IP ranges, in CIDR format, to remove from\n              dynamic advertisement on this {0}. Each IP range in the list must\n              exist in the current set of custom advertisements. This field can\n              only be specified in custom advertisement mode.'.format(resource_str))