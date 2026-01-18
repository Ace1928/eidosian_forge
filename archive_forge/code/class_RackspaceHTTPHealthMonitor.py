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
class RackspaceHTTPHealthMonitor(RackspaceHealthMonitor):
    """
    A HTTP health monitor adds extra features to a Rackspace health monitor.

    :param path: the HTTP path to monitor.
    :type path: ``str``

    :param body_regex: Regular expression used to evaluate the body of
                       the HTTP response.
    :type body_regex: ``str``

    :param status_regex: Regular expression used to evaluate the HTTP
                         status code of the response.
    :type status_regex: ``str``
    """

    def __init__(self, type, delay, timeout, attempts_before_deactivation, path, body_regex, status_regex):
        super().__init__(type, delay, timeout, attempts_before_deactivation)
        self.path = path
        self.body_regex = body_regex
        self.status_regex = status_regex

    def __repr__(self):
        return '<RackspaceHTTPHealthMonitor: type=%s, delay=%d, timeout=%d, attempts_before_deactivation=%d, path=%s, body_regex=%s, status_regex=%s>' % (self.type, self.delay, self.timeout, self.attempts_before_deactivation, self.path, self.body_regex, self.status_regex)

    def _to_dict(self):
        super_dict = super()._to_dict()
        super_dict['path'] = self.path
        super_dict['statusRegex'] = self.status_regex
        if self.body_regex:
            super_dict['bodyRegex'] = self.body_regex
        return super_dict