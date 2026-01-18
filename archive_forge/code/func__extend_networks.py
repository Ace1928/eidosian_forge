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
def _extend_networks(self, networks):
    """Method adds same networks with replaced name on network id.

        This method is used only for resolving attributes.
        """
    nets = copy.deepcopy(networks)
    client_plugin = self.client_plugin('neutron')
    for key in list(nets.keys()):
        try:
            net_id = client_plugin.find_resourceid_by_name_or_id(client_plugin.RES_TYPE_NETWORK, key)
        except Exception as ex:
            if client_plugin.is_not_found(ex) or client_plugin.is_no_unique(ex):
                net_id = None
            else:
                raise
        if net_id:
            nets[net_id] = nets[key]
    return nets