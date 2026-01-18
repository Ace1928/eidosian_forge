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
class RackspaceHealthMonitor:
    """
    :param type: type of load balancer.  currently CONNECT (connection
                 monitoring), HTTP, HTTPS (connection and HTTP
                 monitoring) are supported.
    :type type: ``str``

    :param delay: minimum seconds to wait before executing the health
                      monitor.  (Must be between 1 and 3600)
    :type delay: ``int``

    :param timeout: maximum seconds to wait when establishing a
                    connection before timing out.  (Must be between 1
                    and 3600)
    :type timeout: ``int``

    :param attempts_before_deactivation: Number of monitor failures
                                         before removing a node from
                                         rotation. (Must be between 1
                                         and 10)
    :type attempts_before_deactivation: ``int``
    """

    def __init__(self, type, delay, timeout, attempts_before_deactivation):
        self.type = type
        self.delay = delay
        self.timeout = timeout
        self.attempts_before_deactivation = attempts_before_deactivation

    def __repr__(self):
        return '<RackspaceHealthMonitor: type=%s, delay=%d, timeout=%d, attempts_before_deactivation=%d>' % (self.type, self.delay, self.timeout, self.attempts_before_deactivation)

    def _to_dict(self):
        return {'type': self.type, 'delay': self.delay, 'timeout': self.timeout, 'attemptsBeforeDeactivation': self.attempts_before_deactivation}