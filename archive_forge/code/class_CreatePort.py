import argparse
from oslo_serialization import jsonutils
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.neutron.v2_0 import dns
from neutronclient.neutron.v2_0.qos import policy as qos_policy
class CreatePort(neutronV20.CreateCommand, UpdatePortSecGroupMixin, UpdateExtraDhcpOptMixin, qos_policy.CreateQosPolicyMixin, UpdatePortAllowedAddressPair):
    """Create a port for a given tenant."""
    resource = 'port'

    def add_known_arguments(self, parser):
        _add_updatable_args(parser)
        parser.add_argument('--admin-state-down', dest='admin_state', action='store_false', help=_('Set admin state up to false.'))
        parser.add_argument('--admin_state_down', dest='admin_state', action='store_false', help=argparse.SUPPRESS)
        parser.add_argument('--mac-address', help=_('MAC address of this port.'))
        parser.add_argument('--mac_address', help=argparse.SUPPRESS)
        parser.add_argument('--vnic-type', metavar='<direct | direct-physical | macvtap | normal | baremetal | smart-nic>', choices=['direct', 'direct-physical', 'macvtap', 'normal', 'baremetal', 'smart-nic'], type=utils.convert_to_lowercase, help=_('VNIC type for this port.'))
        parser.add_argument('--vnic_type', choices=['direct', 'direct-physical', 'macvtap', 'normal', 'baremetal', 'smart-nic'], type=utils.convert_to_lowercase, help=argparse.SUPPRESS)
        parser.add_argument('--binding-profile', help=_('Custom data to be passed as binding:profile.'))
        parser.add_argument('--binding_profile', help=argparse.SUPPRESS)
        self.add_arguments_secgroup(parser)
        self.add_arguments_extradhcpopt(parser)
        self.add_arguments_qos_policy(parser)
        self.add_arguments_allowedaddresspairs(parser)
        parser.add_argument('network_id', metavar='NETWORK', help=_('ID or name of the network this port belongs to.'))
        dns.add_dns_argument_create(parser, self.resource, 'name')

    def args2body(self, parsed_args):
        client = self.get_client()
        _network_id = neutronV20.find_resourceid_by_name_or_id(client, 'network', parsed_args.network_id)
        body = {'admin_state_up': parsed_args.admin_state, 'network_id': _network_id}
        _updatable_args2body(parsed_args, body, client)
        neutronV20.update_dict(parsed_args, body, ['mac_address', 'tenant_id'])
        if parsed_args.vnic_type:
            body['binding:vnic_type'] = parsed_args.vnic_type
        if parsed_args.binding_profile:
            body['binding:profile'] = jsonutils.loads(parsed_args.binding_profile)
        self.args2body_secgroup(parsed_args, body)
        self.args2body_extradhcpopt(parsed_args, body)
        self.args2body_qos_policy(parsed_args, body)
        self.args2body_allowedaddresspairs(parsed_args, body)
        dns.args2body_dns_create(parsed_args, body, 'name')
        return {'port': body}