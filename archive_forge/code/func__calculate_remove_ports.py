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
def _calculate_remove_ports(self, old_nets, new_nets, ifaces):
    remove_ports = []
    not_updated_nets = []
    if not old_nets:
        for iface in ifaces:
            remove_ports.append(iface.port_id)
    elif self._str_network(old_nets):
        remove_ports = [iface.port_id for iface in ifaces or []]
    else:
        not_updated_nets = self._exclude_not_updated_networks(old_nets, new_nets, ifaces)
        inter_port_data = self._data_get_ports()
        inter_port_ids = [p['id'] for p in inter_port_data]
        for net in old_nets:
            port_id = net.get(self.NETWORK_PORT)
            if port_id:
                remove_ports.append(port_id)
                if port_id in inter_port_ids:
                    self._delete_internal_port(port_id)
            if net.get(self.NETWORK_FLOATING_IP):
                self._floating_ip_disassociate(net.get(self.NETWORK_FLOATING_IP))
    return (remove_ports, not_updated_nets)