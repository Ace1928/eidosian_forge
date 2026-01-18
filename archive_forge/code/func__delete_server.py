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
def _delete_server(self, server, wait=False, timeout=180, delete_ips=False, delete_ip_retry=1):
    if not server:
        return False
    if delete_ips and self._has_floating_ips() and server['addresses']:
        self._delete_server_floating_ips(server, delete_ip_retry)
    try:
        self.compute.delete_server(server)
    except exceptions.ResourceNotFound:
        return False
    except Exception:
        raise
    if not wait:
        return True
    if not isinstance(server, _server.Server):
        server = _server.Server(id=server['id'])
    self.compute.wait_for_delete(server, wait=timeout)
    return True