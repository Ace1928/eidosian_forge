from datetime import datetime
from libcloud.utils.py3 import httplib
from libcloud.utils.misc import reverse_dict
from libcloud.common.base import JsonResponse, PollingConnection
from libcloud.common.types import LibcloudError
from libcloud.common.openstack import OpenStackDriverMixin
from libcloud.common.rackspace import AUTH_URL
from libcloud.loadbalancer.base import DEFAULT_ALGORITHM, Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State, MemberCondition
from libcloud.compute.drivers.rackspace import RackspaceConnection
def _get_updated_balancer(self, balancer):
    """
        Updating a balancer's attributes puts a balancer into
        'PENDING_UPDATE' status.  Wait until the balancer is
        back in 'ACTIVE' status and then return the individual
        balancer details call.
        """
    resp = self.connection.async_request(action='/loadbalancers/%s' % balancer.id, method='GET')
    return self._to_balancer(resp.object['loadBalancer'])