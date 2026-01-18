from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.instances import flags as instances_flags
from googlecloudsdk.command_lib.compute.resource_policies import flags as maintenance_flags
from googlecloudsdk.command_lib.util.args import labels_util
def AddBulkCreateNetworkingArgs(parser, support_no_address=False, support_network_queue_count=False, support_per_interface_stack_type=False, support_ipv6_only=False):
    """Adds Networking Args for Bulk Create Command."""
    multiple_network_interface_cards_spec = {'network': str, 'subnet': str}

    def ValidateNetworkTier(network_tier_input):
        network_tier = network_tier_input.upper()
        if network_tier in constants.NETWORK_TIER_CHOICES_FOR_INSTANCE:
            return network_tier
        else:
            raise exceptions.InvalidArgumentException('--network-interface', 'Invalid value for network-tier')
    multiple_network_interface_cards_spec['network-tier'] = ValidateNetworkTier
    multiple_network_interface_cards_spec['nic-type'] = instances_flags.ValidateNetworkInterfaceNicType
    network_interface_help = '      Adds a network interface to the instance. Mutually exclusive with any\n      of these flags: *--network*, *--network-tier*, *--no-address*, *--subnet*,\n      *--stack-type*. This flag can be repeated to specify multiple network\n      interfaces.\n\n      *network*::: Specifies the network that the interface will be part of.\n      If subnet is also specified it must be subnetwork of this network. If\n      neither is specified, this defaults to the "default" network.\n\n      *network-tier*::: Specifies the network tier of the interface.\n      ``NETWORK_TIER\'\' must be one of: `PREMIUM`, `STANDARD`. The default\n      value is `PREMIUM`.\n\n      *subnet*::: Specifies the subnet that the interface will be part of.\n      If network key is also specified this must be a subnetwork of the\n      specified network.\n\n      *nic-type*::: Specifies the  Network Interface Controller (NIC) type for\n      the interface. ``NIC_TYPE\'\' must be one of: `GVNIC`, `VIRTIO_NET`.\n  '
    if support_no_address:
        multiple_network_interface_cards_spec['no-address'] = None
        network_interface_help += '\n      *no-address*::: If specified the interface will have no external IP.\n      If not specified instances will get ephemeral IPs.\n      '
    if support_network_queue_count:
        multiple_network_interface_cards_spec['queue-count'] = int
        network_interface_help += "\n      *queue-count*::: Specifies the networking queue count for this interface.\n      Both Rx and Tx queues will be set to this number. If it's not specified, a\n      default queue count will be assigned. See\n      https://cloud.google.com/compute/docs/network-bandwidth#rx-tx for\n      more details.\n    "
    if support_per_interface_stack_type:
        multiple_network_interface_cards_spec['stack-type'] = instances_flags.ValidateNetworkInterfaceStackType if support_ipv6_only else instances_flags.ValidateNetworkInterfaceStackTypeIpv6OnlyNotSupported
        stack_types = '`IPV4_ONLY`, `IPV4_IPV6`, `IPV6_ONLY`' if support_ipv6_only else '`IPV4_ONLY`, `IPV4_IPV6`'
        network_interface_help += f"\n      *stack-type*::: Specifies whether IPv6 is enabled on the interface.\n      ``STACK_TYPE'' must be one of: {stack_types}.\n      The default value is `IPV4_ONLY`.\n    "
    parser.add_argument('--network-interface', type=arg_parsers.ArgDict(spec=multiple_network_interface_cards_spec, allow_key_only=True), action='append', metavar='PROPERTY=VALUE', help=network_interface_help)