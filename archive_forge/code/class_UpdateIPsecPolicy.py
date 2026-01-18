import argparse
from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronv20
from neutronclient.neutron.v2_0.vpn import utils as vpn_utils
class UpdateIPsecPolicy(neutronv20.UpdateCommand):
    """Update a given IPsec policy."""
    resource = 'ipsecpolicy'
    help_resource = 'IPsec policy'

    def add_known_arguments(self, parser):
        parser.add_argument('--name', help=_('Updated name of the IPsec policy.'))
        add_common_args(parser, is_create=False)

    def args2body(self, parsed_args):
        return {'ipsecpolicy': parse_common_args2body(parsed_args, body={})}