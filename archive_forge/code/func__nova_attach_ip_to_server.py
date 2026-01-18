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
def _nova_attach_ip_to_server(self, server_id, floating_ip_id, fixed_address=None):
    f_ip = self.get_floating_ip(id=floating_ip_id)
    if f_ip is None:
        raise exceptions.SDKException('unable to find floating IP {0}'.format(floating_ip_id))
    error_message = 'Error attaching IP {ip} to instance {id}'.format(ip=floating_ip_id, id=server_id)
    body = {'address': f_ip['floating_ip_address']}
    if fixed_address:
        body['fixed_address'] = fixed_address
    return proxy._json_response(self.compute.post('/servers/{server_id}/action'.format(server_id=server_id), json=dict(addFloatingIp=body)), error_message=error_message)