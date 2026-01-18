from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddStackTypeArg(parser):
    parser.add_argument('--stack-type', choices={'IPV4_ONLY': 'The network interface will be assigned IPv4 addresses.', 'IPV4_IPV6': 'The network interface can have both IPv4 and IPv6 addresses.'}, type=arg_utils.ChoiceToEnumName, help='The stack type for the default network interface. Determines if IPv6 is enabled on the default network interface.')