from openstack.cloud import _utils
from openstack.cloud import exc
from openstack import exceptions
from openstack.network.v2._proxy import Proxy
from openstack import proxy
from openstack import utils
def _has_secgroups(self):
    if not self.secgroup_source:
        return False
    else:
        return self.secgroup_source.lower() in ('nova', 'neutron')