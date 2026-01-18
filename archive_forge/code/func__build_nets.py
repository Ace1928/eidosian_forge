import copy
import shlex
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine.clients import progress
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources.openstack.nova import server_network_mixin
from heat.engine import support
from heat.engine import translation
def _build_nets(self, networks):
    nics = self._build_nics(networks)
    for nic in nics:
        net_id = nic.pop('net-id', None)
        if net_id:
            nic[self.NETWORK_ID] = net_id
        port_id = nic.pop('port-id', None)
        if port_id:
            nic[self.NETWORK_PORT] = port_id
    return nics