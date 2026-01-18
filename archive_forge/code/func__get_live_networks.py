import copy
import ipaddress
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import uuidutils
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine.clients import progress
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.neutron import port as neutron_port
from heat.engine.resources.openstack.neutron import subnet
from heat.engine.resources.openstack.nova import server_network_mixin
from heat.engine.resources import scheduler_hints as sh
from heat.engine.resources import server_base
from heat.engine import support
from heat.engine import translation
from heat.rpc import api as rpc_api
def _get_live_networks(self, server, props):
    reality_nets = self._get_server_addresses(server, extend_networks=False)
    reality_net_ids = {}
    client_plugin = self.client_plugin('neutron')
    for net_key in reality_nets:
        try:
            net_id = client_plugin.find_resourceid_by_name_or_id(client_plugin.RES_TYPE_NETWORK, net_key)
        except Exception as ex:
            if client_plugin.is_not_found(ex) or client_plugin.is_no_unique(ex):
                net_id = None
            else:
                raise
        if net_id:
            reality_net_ids[net_id] = reality_nets.get(net_key)
    resource_nets = props.get(self.NETWORKS)
    result_nets = []
    for net in resource_nets or []:
        net_id = self._get_network_id(net)
        if reality_net_ids.get(net_id):
            for idx, address in enumerate(reality_net_ids.get(net_id)):
                if address['addr'] == net[self.NETWORK_FIXED_IP]:
                    result_nets.append(net)
                    reality_net_ids.get(net_id).pop(idx)
                    break
    for key, value in reality_nets.items():
        for address in reality_nets[key]:
            new_net = {self.NETWORK_ID: key, self.NETWORK_FIXED_IP: address['addr']}
            if address['port'] not in [port['id'] for port in self._data_get_ports()]:
                new_net.update({self.NETWORK_PORT: address['port']})
            result_nets.append(new_net)
    return result_nets