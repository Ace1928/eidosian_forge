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
def _neutron_detach_ip_from_server(self, server_id, floating_ip_id):
    f_ip = self.get_floating_ip(id=floating_ip_id)
    if f_ip is None or not bool(f_ip.port_id):
        return False
    try:
        self.network.update_ip(floating_ip_id, port_id=None)
    except exceptions.SDKException:
        raise exceptions.SDKException('Error detaching IP {ip} from server {server_id}'.format(ip=floating_ip_id, server_id=server_id))
    return True