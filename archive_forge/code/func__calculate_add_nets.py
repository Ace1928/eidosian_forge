import itertools
import eventlet
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import netutils
import tenacity
from heat.common import exception
from heat.common.i18n import _
from heat.engine import resource
from heat.engine.resources.openstack.neutron import port as neutron_port
def _calculate_add_nets(self, new_nets, not_updated_nets, security_groups):
    add_nets = []
    if not new_nets and (not not_updated_nets):
        handler_kwargs = {'port_id': None, 'net_id': None, 'fip': None}
        if security_groups:
            sec_uuids = self.client_plugin('neutron').get_secgroup_uuids(security_groups)
            handler_kwargs['security_groups'] = sec_uuids
        add_nets.append(handler_kwargs)
    else:
        for idx, net in enumerate(new_nets):
            handler_kwargs = {'port_id': None, 'net_id': None, 'fip': None}
            if net.get(self.NETWORK_PORT):
                handler_kwargs['port_id'] = net.get(self.NETWORK_PORT)
            elif net.get(self.NETWORK_SUBNET):
                handler_kwargs['port_id'] = self._create_internal_port(net, idx, security_groups)
            if not handler_kwargs['port_id']:
                handler_kwargs['net_id'] = self._get_network_id(net)
                if security_groups:
                    sec_uuids = self.client_plugin('neutron').get_secgroup_uuids(security_groups)
                    handler_kwargs['security_groups'] = sec_uuids
            if handler_kwargs['net_id']:
                handler_kwargs['fip'] = net.get('fixed_ip')
            floating_ip = net.get(self.NETWORK_FLOATING_IP)
            if floating_ip:
                flip_associate = {'port_id': handler_kwargs.get('port_id')}
                if net.get('fixed_ip'):
                    flip_associate['fixed_ip_address'] = net.get('fixed_ip')
                self.update_floating_ip_association(floating_ip, flip_associate)
            add_nets.append(handler_kwargs)
    return add_nets