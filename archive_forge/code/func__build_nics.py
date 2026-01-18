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
def _build_nics(self, networks, security_groups=None):
    if not networks:
        return None
    str_network = self._str_network(networks)
    if str_network:
        return str_network
    nics = []
    for idx, net in enumerate(networks):
        self._validate_belonging_subnet_to_net(net)
        nic_info = {'net-id': self._get_network_id(net)}
        if net.get(self.NETWORK_PORT):
            nic_info['port-id'] = net[self.NETWORK_PORT]
        elif net.get(self.NETWORK_SUBNET):
            nic_info['port-id'] = self._create_internal_port(net, idx, security_groups)
        if not nic_info.get('port-id'):
            if net.get(self.NETWORK_FIXED_IP):
                ip = net[self.NETWORK_FIXED_IP]
                if netutils.is_valid_ipv6(ip):
                    nic_info['v6-fixed-ip'] = ip
                else:
                    nic_info['v4-fixed-ip'] = ip
        if net.get(self.NETWORK_FLOATING_IP) and nic_info.get('port-id'):
            floating_ip_data = {'port_id': nic_info['port-id']}
            if net.get(self.NETWORK_FIXED_IP):
                floating_ip_data.update({'fixed_ip_address': net.get(self.NETWORK_FIXED_IP)})
            self._floating_ip_neutron_associate(net.get(self.NETWORK_FLOATING_IP), floating_ip_data)
        if net.get(self.NIC_TAG):
            nic_info[self.NIC_TAG] = net.get(self.NIC_TAG)
        nics.append(nic_info)
    return nics