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
def _DeleteLoadBalancer(self, method, url, body, headers):
    params = {'LoadBalancerId': '15229f88562-cn-hangzhou-dg-a01'}
    self.assertUrlContainsQueryParams(url, params)
    body = self.fixtures.load('delete_load_balancer.xml')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])