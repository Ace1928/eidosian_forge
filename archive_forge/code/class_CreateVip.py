from neutronclient._i18n import _
from neutronclient.neutron import v2_0 as neutronV20
class CreateVip(neutronV20.CreateCommand):
    """Create a vip."""
    resource = 'vip'

    def add_known_arguments(self, parser):
        parser.add_argument('pool_id', metavar='POOL', help=_('ID or name of the pool to which this vip belongs.'))
        parser.add_argument('--address', help=_('IP address of the vip.'))
        parser.add_argument('--admin-state-down', dest='admin_state', action='store_false', help=_('Set admin state up to false.'))
        parser.add_argument('--connection-limit', help=_('The maximum number of connections per second allowed for the vip. Valid values: a positive integer or -1 for unlimited (default).'))
        parser.add_argument('--description', help=_('Description of the vip to be created.'))
        parser.add_argument('--name', required=True, help=_('Name of the vip to be created.'))
        parser.add_argument('--protocol-port', required=True, help=_('TCP port on which to listen for client traffic that is associated with the vip address.'))
        parser.add_argument('--protocol', required=True, choices=['TCP', 'HTTP', 'HTTPS'], help=_('Protocol for balancing.'))
        parser.add_argument('--subnet-id', metavar='SUBNET', required=True, help=_('The subnet on which to allocate the vip address.'))

    def args2body(self, parsed_args):
        _pool_id = neutronV20.find_resourceid_by_name_or_id(self.get_client(), 'pool', parsed_args.pool_id)
        _subnet_id = neutronV20.find_resourceid_by_name_or_id(self.get_client(), 'subnet', parsed_args.subnet_id)
        body = {'pool_id': _pool_id, 'admin_state_up': parsed_args.admin_state, 'subnet_id': _subnet_id}
        neutronV20.update_dict(parsed_args, body, ['address', 'connection_limit', 'description', 'name', 'protocol_port', 'protocol', 'tenant_id'])
        return {self.resource: body}