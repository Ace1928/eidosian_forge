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
def _extend_addresses(self, container):
    """Method adds network name to list of addresses.

        This method is used only for resolving attributes.
        """
    nets = self.neutron().list_networks()['networks']
    id_name_mapping_on_network = {net['id']: net['name'] for net in nets}
    addresses = copy.deepcopy(container.addresses)
    for net_uuid in container.addresses or {}:
        addr_list = addresses[net_uuid]
        net_name = id_name_mapping_on_network.get(net_uuid)
        if not net_name:
            continue
        addresses.setdefault(net_name, [])
        addresses[net_name] += addr_list
    return addresses