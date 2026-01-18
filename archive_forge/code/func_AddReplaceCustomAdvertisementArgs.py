from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddReplaceCustomAdvertisementArgs(parser, resource_str):
    """Adds common arguments for replacing custom advertisements."""
    parser.add_argument('--advertisement-mode', choices=_MODE_CHOICES, type=lambda mode: mode.upper(), metavar='MODE', help='The new advertisement mode for this {0}.'.format(resource_str))
    parser.add_argument('--set-advertisement-groups', type=arg_parsers.ArgList(choices=_GROUP_CHOICES, element_type=lambda group: group.upper()), metavar='GROUP', help='The list of pre-defined groups of IP ranges to dynamically\n              advertise on this {0}. This list can only be specified in\n              custom advertisement mode.'.format(resource_str))
    parser.add_argument('--set-advertisement-ranges', type=arg_parsers.ArgDict(allow_key_only=True), metavar='CIDR_RANGE=DESC', help='The list of individual IP ranges, in CIDR format, to dynamically\n              advertise on this {0}. Each IP range can (optionally) be given a\n              text description DESC. For example, to advertise a specific range,\n              use `--set-advertisement-ranges=192.168.10.0/24`.  To store a\n              description with the range, use\n              `--set-advertisement-ranges=192.168.10.0/24=my-networks`. This\n              list can only be specified in custom advertisement mode.'.format(resource_str))