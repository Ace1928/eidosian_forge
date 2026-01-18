import ipaddress
import time
import warnings
from openstack.cloud import _utils
from openstack.cloud import exc
from openstack.cloud import meta
from openstack import exceptions
from openstack.network.v2._proxy import Proxy
from openstack import proxy
from openstack import utils
from openstack import warnings as os_warnings
def _get_floating_network_id(self):
    networks = self.get_external_ipv4_floating_networks()
    if networks:
        floating_network_id = networks[0]['id']
    else:
        floating_network = self._find_floating_network_by_router()
        if floating_network:
            floating_network_id = floating_network
        else:
            raise exceptions.NotFoundException('unable to find an external network')
    return floating_network_id