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
def _expand_server(self, server, detailed, bare):
    if bare or not server:
        return server
    elif detailed:
        return meta.get_hostvars_from_server(self, server)
    else:
        return meta.add_server_interfaces(self, server)