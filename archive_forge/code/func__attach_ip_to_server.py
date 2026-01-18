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
def _attach_ip_to_server(self, server, floating_ip, fixed_address=None, wait=False, timeout=60, skip_attach=False, nat_destination=None):
    """Attach a floating IP to a server.

        :param server: Server dict
        :param floating_ip: Floating IP dict to attach
        :param fixed_address: (optional) fixed address to which attach the
                              floating IP to.
        :param wait: (optional) Wait for the address to appear as assigned
                     to the server. Defaults to False.
        :param timeout: (optional) Seconds to wait, defaults to 60.
                        See the ``wait`` parameter.
        :param skip_attach: (optional) Skip the actual attach and just do
                            the wait. Defaults to False.
        :param nat_destination: The fixed network the server's port for the
                                FIP to attach to will come from.

        :returns: The server ``openstack.compute.v2.server.Server``
        :raises: :class:`~openstack.exceptions.SDKException` on operation
            error.
        """
    ext_ip = meta.get_server_ip(server, ext_tag='floating', public=True)
    if not ext_ip and floating_ip['port_id']:
        server = self.get_server_by_id(server['id'])
        ext_ip = meta.get_server_ip(server, ext_tag='floating', public=True)
    if ext_ip == floating_ip['floating_ip_address']:
        return server
    if self._use_neutron_floating():
        if not skip_attach:
            try:
                self._neutron_attach_ip_to_server(server=server, floating_ip=floating_ip, fixed_address=fixed_address, nat_destination=nat_destination)
            except exceptions.NotFoundException as e:
                self.log.debug("Something went wrong talking to neutron API: '%(msg)s'. Trying with Nova.", {'msg': str(e)})
    else:
        self._nova_attach_ip_to_server(server_id=server['id'], floating_ip_id=floating_ip['id'], fixed_address=fixed_address)
    if wait:
        server_id = server['id']
        for _ in utils.iterate_timeout(timeout, 'Timeout waiting for the floating IP to be attached.', wait=min(5, timeout)):
            server = self.get_server_by_id(server_id)
            ext_ip = meta.get_server_ip(server, ext_tag='floating', public=True)
            if ext_ip == floating_ip['floating_ip_address']:
                return server
    return server