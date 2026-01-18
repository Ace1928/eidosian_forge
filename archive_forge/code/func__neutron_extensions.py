from openstack.cloud import _utils
from openstack.cloud import exc
from openstack import exceptions
from openstack.network.v2._proxy import Proxy
def _neutron_extensions(self):
    extensions = set()
    for extension in self.network.extensions():
        extensions.add(extension['alias'])
    return extensions