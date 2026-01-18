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
def _needs_floating_ip(self, server, nat_destination):
    """Figure out if auto_ip should add a floating ip to this server.

        If the server has a floating ip it does not need another one.

        If the server does not have a fixed ip address it does not need a
        floating ip.

        If self.private then the server does not need a floating ip.

        If the cloud runs nova, and the server has a private address and not a
        public address, then the server needs a floating ip.

        If the server has a fixed ip address and no floating ip address and the
        cloud has a network from which floating IPs come that is connected via
        a router to the network from which the fixed ip address came,
        then the server needs a floating ip.

        If the server has a fixed ip address and no floating ip address and the
        cloud does not have a network from which floating ips come, or it has
        one but that network is not connected to the network from which
        the server's fixed ip address came via a router, then the
        server does not need a floating ip.
        """
    if not self._has_floating_ips():
        return False
    if server['addresses'] is None:
        server = self.compute.get_server(server)
    if server['public_v4'] or any([any([address['OS-EXT-IPS:type'] == 'floating' for address in addresses]) for addresses in (server['addresses'] or {}).values()]):
        return False
    if not server['private_v4'] and (not any([any([address['OS-EXT-IPS:type'] == 'fixed' for address in addresses]) for addresses in (server['addresses'] or {}).values()])):
        return False
    if self.private:
        return False
    if not self.has_service('network'):
        return True
    try:
        self._get_floating_network_id()
    except exceptions.SDKException:
        return False
    port_obj, fixed_ip_address = self._nat_destination_port(server, nat_destination=nat_destination)
    if not port_obj or not fixed_ip_address:
        return False
    return True