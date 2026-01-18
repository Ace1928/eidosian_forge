from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def _AddSubnetworkArgs(parser, for_create):
    """Adds a mutually exclusive group to specify subnet options."""
    subnetwork = parser.add_mutually_exclusive_group(required=for_create)
    subnetwork.add_argument('--nat-all-subnet-ip-ranges', help=textwrap.dedent('          Allow all IP ranges of all subnetworks in the region, including\n          primary and secondary ranges, to use NAT.'), action='store_const', dest='subnet_option', const=SubnetOption.ALL_RANGES, default=SubnetOption.CUSTOM_RANGES)
    subnetwork.add_argument('--nat-primary-subnet-ip-ranges', help=textwrap.dedent('          Allow only primary IP ranges of all subnetworks in the region to use\n          NAT.'), action='store_const', dest='subnet_option', const=SubnetOption.PRIMARY_RANGES, default=SubnetOption.CUSTOM_RANGES)
    custom_subnet_help_text = '    List of subnetwork primary and secondary IP ranges to be allowed to\n    use NAT.\n\n    * `SUBNETWORK:ALL` - specifying a subnetwork name with ALL includes the\n    primary range and all secondary ranges of the subnet.\n    * `SUBNETWORK` - including a subnetwork name includes only the primary\n    subnet range of the subnetwork.\n    * `SUBNETWORK:RANGE_NAME` - specifying a subnetwork and secondary range\n    name includes only that secondary range. It does not include the\n    primary range of the subnet.\n    '
    subnetwork.add_argument('--nat-custom-subnet-ip-ranges', metavar='SUBNETWORK[:RANGE_NAME|:ALL]', help=custom_subnet_help_text, type=arg_parsers.ArgList(min_length=1))