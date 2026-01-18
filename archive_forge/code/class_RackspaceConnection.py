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
class RackspaceConnection(RackspaceConnection, PollingConnection):
    responseCls = RackspaceResponse
    auth_url = AUTH_URL
    poll_interval = 2
    timeout = 80
    cache_busting = True

    def request(self, action, params=None, data='', headers=None, method='GET'):
        if not headers:
            headers = {}
        if not params:
            params = {}
        if method in ('POST', 'PUT'):
            headers['Content-Type'] = 'application/json'
        return super().request(action=action, params=params, data=data, method=method, headers=headers)

    def get_poll_request_kwargs(self, response, context, request_kwargs):
        return {'action': request_kwargs['action'], 'method': 'GET'}

    def has_completed(self, response):
        state = response.object['loadBalancer']['status']
        if state == 'ERROR':
            raise LibcloudError('Load balancer entered an ERROR state.', driver=self.driver)
        return state == 'ACTIVE'

    def encode_data(self, data):
        return data