import argparse
from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronv20
from neutronclient.neutron.v2_0.vpn import utils as vpn_utils
class CreateIPsecPolicy(neutronv20.CreateCommand):
    """Create an IPsec policy."""
    resource = 'ipsecpolicy'
    help_resource = 'IPsec policy'

    def add_known_arguments(self, parser):
        parser.add_argument('name', metavar='NAME', help=_('Name of the IPsec policy.'))
        add_common_args(parser)

    def args2body(self, parsed_args):
        return {'ipsecpolicy': parse_common_args2body(parsed_args, body={})}