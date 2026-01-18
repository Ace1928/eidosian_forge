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
def _nova_available_floating_ips(self, pool=None):
    """Get available floating IPs from a floating IP pool.

        Return a list of available floating IPs or allocate a new one and
        return it in a list of 1 element.

        :param pool: Nova floating IP pool name.

        :returns: a list of floating IP addresses.
        :raises: :class:`~openstack.exceptions.BadRequestException` if a
            floating IP pool is not specified and cannot be found.
        """
    with _utils.openstacksdk_exceptions('Unable to create floating IP in pool {pool}'.format(pool=pool)):
        if pool is None:
            pools = self.list_floating_ip_pools()
            if not pools:
                raise exceptions.NotFoundException('unable to find a floating ip pool')
            pool = pools[0]['name']
        filters = {'instance_id': None, 'pool': pool}
        floating_ips = self._nova_list_floating_ips()
        available_ips = _utils._filter_list(floating_ips, name_or_id=None, filters=filters)
        if available_ips:
            return available_ips
        f_ip = self._nova_create_floating_ip(pool=pool)
        return [f_ip]