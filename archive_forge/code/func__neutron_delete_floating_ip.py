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
def _neutron_delete_floating_ip(self, floating_ip_id):
    try:
        self.network.delete_ip(floating_ip_id, ignore_missing=False)
    except exceptions.ResourceNotFound:
        return False
    return True