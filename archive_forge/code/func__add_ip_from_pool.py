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
def _add_ip_from_pool(self, server, network, fixed_address=None, reuse=True, wait=False, timeout=60, nat_destination=None):
    """Add a floating IP to a server from a given pool

        This method reuses available IPs, when possible, or allocate new IPs
        to the current tenant.
        The floating IP is attached to the given fixed address or to the
        first server port/fixed address

        :param server: Server dict
        :param network: Name or ID of the network.
        :param fixed_address: a fixed address
        :param reuse: Try to reuse existing ips. Defaults to True.
        :param wait: (optional) Wait for the address to appear as assigned
                     to the server. Defaults to False.
        :param timeout: (optional) Seconds to wait, defaults to 60.
                        See the ``wait`` parameter.
        :param nat_destination: (optional) the name of the network of the
                                port to associate with the floating ip.

        :returns: the updated server ``openstack.compute.v2.server.Server``
        """
    if reuse:
        f_ip = self.available_floating_ip(network=network)
    else:
        start_time = time.time()
        f_ip = self.create_floating_ip(server=server, network=network, nat_destination=nat_destination, fixed_address=fixed_address, wait=wait, timeout=timeout)
        timeout = timeout - (time.time() - start_time)
        server = self.get_server(server.id)
    return self._attach_ip_to_server(server=server, floating_ip=f_ip, fixed_address=fixed_address, wait=wait, timeout=timeout, nat_destination=nat_destination)