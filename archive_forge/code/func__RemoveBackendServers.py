import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node
from libcloud.test.secrets import LB_SLB_PARAMS
from libcloud.compute.types import NodeState
from libcloud.loadbalancer.base import Member, Algorithm
from libcloud.loadbalancer.types import State
from libcloud.test.file_fixtures import LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.slb import (
def _RemoveBackendServers(self, method, url, body, headers):
    _id = self.test.member.id
    servers_json = '["%s"]' % _id
    params = {'LoadBalancerId': self.test.balancer.id, 'BackendServers': servers_json}
    self.assertUrlContainsQueryParams(url, params)
    body = self.fixtures.load('add_backend_servers.xml')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])