from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronv20
class CreateFirewall(neutronv20.CreateCommand):
    """Create a firewall."""
    resource = 'firewall'

    def add_known_arguments(self, parser):
        add_common_args(parser)
        parser.add_argument('policy', metavar='POLICY', help=_('ID or name of the firewall policy associated to this firewall.'))
        parser.add_argument('--admin-state-down', dest='admin_state', action='store_false', help=_('Set admin state up to false.'))

    def args2body(self, parsed_args):
        body = parse_common_args(self.get_client(), parsed_args)
        neutronv20.update_dict(parsed_args, body, ['tenant_id'])
        body['admin_state_up'] = parsed_args.admin_state
        return {self.resource: body}