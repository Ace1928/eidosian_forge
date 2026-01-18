from oslo_log import log as logging
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
from os_win.utils.network import networkutils
def _get_network_iface_index(self, network_name):
    if self._net_if_indexes.get(network_name):
        return self._net_if_indexes[network_name]
    description = self._utils.get_vswitch_external_network_name(network_name)
    networks = self._scimv2.MSFT_NetAdapter(InterfaceDescription=description)
    if not networks:
        raise exceptions.NotFound(resource=network_name)
    self._net_if_indexes[network_name] = networks[0].InterfaceIndex
    return networks[0].InterfaceIndex