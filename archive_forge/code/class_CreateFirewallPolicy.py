import argparse
from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronv20
class CreateFirewallPolicy(neutronv20.CreateCommand):
    """Create a firewall policy."""
    resource = 'firewall_policy'

    def add_known_arguments(self, parser):
        parser.add_argument('name', metavar='NAME', help=_('Name for the firewall policy.'))
        parser.add_argument('--shared', action='store_true', help=_('Create a shared policy.'), default=argparse.SUPPRESS)
        parser.add_argument('--audited', action='store_true', help=_('Sets audited to True.'), default=argparse.SUPPRESS)
        add_common_args(parser)

    def args2body(self, parsed_args):
        return parse_common_args(self.get_client(), parsed_args)