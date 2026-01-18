from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddAddressArgs(parser):
    """Adds --address and --no-address mutex arguments to the parser."""
    addresses = parser.add_mutually_exclusive_group()
    addresses.add_argument('--address', type=str, help='\n        Assigns the given external address to the network interface. The\n        address might be an IP address or the name or URI of an address\n        resource. Specifying an empty string will assign an ephemeral IP.\n        Mutually exclusive with no-address. If neither key is present the\n        network interface will get an ephemeral IP.\n      ')
    addresses.add_argument('--no-address', action='store_true', help='\n        If specified the network interface will have no external IP.\n        Mutually exclusive with address. If neither key is present the network\n        interfaces will get an ephemeral IP.\n      ')