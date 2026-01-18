from openstack.cloud import _utils
from openstack.cloud import exc
from openstack import exceptions
from openstack.network.v2._proxy import Proxy
from openstack import proxy
from openstack import utils
def _normalize_secgroups(self, groups):
    """Normalize the structure of security groups

        This makes security group dicts, as returned from nova, look like the
        security group dicts as returned from neutron. This does not make them
        look exactly the same, but it's pretty close.

        :param list groups: A list of security group dicts.

        :returns: A list of normalized dicts.
        """
    ret = []
    for group in groups:
        ret.append(self._normalize_secgroup(group))
    return ret