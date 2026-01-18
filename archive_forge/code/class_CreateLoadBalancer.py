from oslo_serialization import jsonutils
from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
class CreateLoadBalancer(neutronV20.CreateCommand):
    """LBaaS v2 Create a loadbalancer."""
    resource = 'loadbalancer'

    def add_known_arguments(self, parser):
        _add_common_args(parser)
        parser.add_argument('--admin-state-down', dest='admin_state', action='store_false', help=_('Set admin state up to false.'))
        parser.add_argument('--provider', help=_('Provider name of the load balancer service.'))
        parser.add_argument('--flavor', help=_('ID or name of the flavor.'))
        parser.add_argument('--vip-address', help=_('VIP address for the load balancer.'))
        parser.add_argument('vip_subnet', metavar='VIP_SUBNET', help=_('Load balancer VIP subnet.'))

    def args2body(self, parsed_args):
        _subnet_id = neutronV20.find_resourceid_by_name_or_id(self.get_client(), 'subnet', parsed_args.vip_subnet)
        body = {'vip_subnet_id': _subnet_id, 'admin_state_up': parsed_args.admin_state}
        if parsed_args.flavor:
            _flavor_id = neutronV20.find_resourceid_by_name_or_id(self.get_client(), 'flavor', parsed_args.flavor)
            body['flavor_id'] = _flavor_id
        neutronV20.update_dict(parsed_args, body, ['provider', 'vip_address', 'tenant_id'])
        _parse_common_args(body, parsed_args)
        return {self.resource: body}