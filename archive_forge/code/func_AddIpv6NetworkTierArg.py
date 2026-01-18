from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddIpv6NetworkTierArg(parser):
    parser.add_argument('--ipv6-network-tier', choices={'PREMIUM': 'High quality, Google-grade network tier.'}, type=arg_utils.ChoiceToEnumName, help='Specifies the IPv6 network tier that will be used to configure the instance network interface IPv6 access config.')