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
def _neutron_available_floating_ips(self, network=None, project_id=None, server=None):
    """Get a floating IP from a network.

        Return a list of available floating IPs or allocate a new one and
        return it in a list of 1 element.

        :param network: A single network name or ID, or a list of them.
        :param server: (server) Server the Floating IP is for

        :returns: a list of floating IP addresses.
        :raises: :class:`~openstack.exceptions.BadRequestException` if an
            external network that meets the specified criteria cannot be found.
        """
    if project_id is None:
        project_id = self.current_project_id
    if network:
        if isinstance(network, str):
            network = [network]
        floating_network_id = None
        for net in network:
            for ext_net in self.get_external_ipv4_floating_networks():
                if net in (ext_net['name'], ext_net['id']):
                    floating_network_id = ext_net['id']
                    break
            if floating_network_id:
                break
        if floating_network_id is None:
            raise exceptions.NotFoundException('unable to find external network {net}'.format(net=network))
    else:
        floating_network_id = self._get_floating_network_id()
    filters = {'port_id': None, 'floating_network_id': floating_network_id, 'project_id': project_id}
    floating_ips = self.list_floating_ips()
    available_ips = _utils._filter_list(floating_ips, name_or_id=None, filters=filters)
    if available_ips:
        return available_ips
    f_ip = self._neutron_create_floating_ip(network_id=floating_network_id, server=server)
    return [f_ip]