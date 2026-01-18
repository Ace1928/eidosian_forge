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
def _has_floating_ips(self):
    if not self._floating_ip_source:
        return False
    else:
        return self._floating_ip_source in ('nova', 'neutron')