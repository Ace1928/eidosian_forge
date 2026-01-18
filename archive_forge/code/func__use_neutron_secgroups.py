from openstack.cloud import _utils
from openstack.cloud import exc
from openstack import exceptions
from openstack.network.v2._proxy import Proxy
from openstack import proxy
from openstack import utils
def _use_neutron_secgroups(self):
    return self.has_service('network') and self.secgroup_source == 'neutron'