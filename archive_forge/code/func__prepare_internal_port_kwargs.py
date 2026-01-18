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
def _prepare_internal_port_kwargs(self, net_data, security_groups=None):
    kwargs = {'network_id': self._get_network_id(net_data)}
    fixed_ip = net_data.get(self.NETWORK_FIXED_IP)
    subnet = net_data.get(self.NETWORK_SUBNET)
    body = {}
    if fixed_ip:
        body['ip_address'] = fixed_ip
    if subnet:
        body['subnet_id'] = subnet
    if body:
        kwargs.update({'fixed_ips': [body]})
    if security_groups:
        sec_uuids = self.client_plugin('neutron').get_secgroup_uuids(security_groups)
        kwargs['security_groups'] = sec_uuids
    extra_props = net_data.get(self.NETWORK_PORT_EXTRA)
    if extra_props is not None:
        specs = extra_props.pop(neutron_port.Port.VALUE_SPECS)
        if specs:
            kwargs.update(specs)
        port_extra_keys = list(neutron_port.Port.EXTRA_PROPERTIES)
        port_extra_keys.remove(neutron_port.Port.ALLOWED_ADDRESS_PAIRS)
        for key in port_extra_keys:
            if extra_props.get(key) is not None:
                kwargs[key] = extra_props.get(key)
        allowed_address_pairs = extra_props.get(neutron_port.Port.ALLOWED_ADDRESS_PAIRS)
        if allowed_address_pairs is not None:
            for pair in allowed_address_pairs:
                if neutron_port.Port.ALLOWED_ADDRESS_PAIR_MAC_ADDRESS in pair and pair.get(neutron_port.Port.ALLOWED_ADDRESS_PAIR_MAC_ADDRESS) is None:
                    del pair[neutron_port.Port.ALLOWED_ADDRESS_PAIR_MAC_ADDRESS]
            port_address_pairs = neutron_port.Port.ALLOWED_ADDRESS_PAIRS
            kwargs[port_address_pairs] = allowed_address_pairs
    return kwargs