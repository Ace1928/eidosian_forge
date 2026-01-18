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
def _encode_server_userdata(self, userdata):
    if hasattr(userdata, 'read'):
        userdata = userdata.read()
    if not isinstance(userdata, bytes):
        if not isinstance(userdata, str):
            raise TypeError("%s can't be encoded" % type(userdata))
        userdata = userdata.encode('utf-8', 'strict')
    return base64.b64encode(userdata).decode('utf-8')