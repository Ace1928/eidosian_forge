from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
class CreateListener(neutronV20.CreateCommand):
    """LBaaS v2 Create a listener."""
    resource = 'listener'

    def add_known_arguments(self, parser):
        _add_common_args(parser)
        parser.add_argument('--admin-state-down', dest='admin_state', action='store_false', help=_('Set admin state up to false.'))
        parser.add_argument('--name', help=_('The name of the listener. At least one of --default-pool or --loadbalancer must be specified.'))
        parser.add_argument('--default-tls-container-ref', dest='default_tls_container_ref', help=_('Default TLS container reference to retrieve TLS information.'))
        parser.add_argument('--sni-container-refs', dest='sni_container_refs', nargs='+', help=_('List of TLS container references for SNI.'))
        parser.add_argument('--loadbalancer', metavar='LOADBALANCER', help=_('ID or name of the load balancer.'))
        parser.add_argument('--protocol', required=True, choices=['TCP', 'HTTP', 'HTTPS', 'TERMINATED_HTTPS'], type=utils.convert_to_uppercase, help=_('Protocol for the listener.'))
        parser.add_argument('--protocol-port', dest='protocol_port', required=True, metavar='PORT', help=_('Protocol port for the listener.'))

    def args2body(self, parsed_args):
        if not parsed_args.loadbalancer and (not parsed_args.default_pool):
            message = _('Either --default-pool or --loadbalancer must be specified.')
            raise exceptions.CommandError(message)
        body = {'protocol': parsed_args.protocol, 'protocol_port': parsed_args.protocol_port, 'admin_state_up': parsed_args.admin_state}
        if parsed_args.loadbalancer:
            loadbalancer_id = _get_loadbalancer_id(self.get_client(), parsed_args.loadbalancer)
            body['loadbalancer_id'] = loadbalancer_id
        neutronV20.update_dict(parsed_args, body, ['default_tls_container_ref', 'sni_container_refs', 'tenant_id'])
        _parse_common_args(body, parsed_args, self.get_client())
        return {self.resource: body}