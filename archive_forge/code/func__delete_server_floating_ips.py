import base64
import functools
import operator
import time
import iso8601
from openstack.cloud import _utils
from openstack.cloud import exc
from openstack.cloud import meta
from openstack.compute.v2._proxy import Proxy
from openstack.compute.v2 import quota_set as _qs
from openstack.compute.v2 import server as _server
from openstack import exceptions
from openstack import utils
def _delete_server_floating_ips(self, server, delete_ip_retry):
    server_floats = meta.find_nova_interfaces(server['addresses'], ext_tag='floating')
    for fip in server_floats:
        try:
            ip = self.get_floating_ip(id=None, filters={'floating_ip_address': fip['addr']})
        except exceptions.NotFoundException:
            continue
        if not ip:
            continue
        deleted = self.delete_floating_ip(ip['id'], retry=delete_ip_retry)
        if not deleted:
            raise exceptions.SDKException('Tried to delete floating ip {floating_ip} associated with server {id} but there was an error deleting it. Not deleting server.'.format(floating_ip=ip['floating_ip_address'], id=server['id']))