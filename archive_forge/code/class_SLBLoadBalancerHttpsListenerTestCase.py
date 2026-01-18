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
class SLBLoadBalancerHttpsListenerTestCase(unittest.TestCase, AssertDictMixin):

    def setUp(self):
        self.listener = SLBLoadBalancerHttpsListener.create(80, 8080, Algorithm.WEIGHTED_ROUND_ROBIN, 1, extra={'StickySession': 'on', 'StickySessionType': 'insert', 'HealthCheck': 'on', 'ServerCertificateId': 'fake-cert1'})

    def test_get_required_params(self):
        expected = {'Action': 'CreateLoadBalancerHTTPSListener', 'ListenerPort': 80, 'BackendServerPort': 8080, 'Scheduler': 'wrr', 'Bandwidth': 1, 'StickySession': 'on', 'HealthCheck': 'on', 'ServerCertificateId': 'fake-cert1'}
        self.assert_dict_equals(expected, self.listener.get_required_params())

    def test_get_optional_params(self):
        expected = {'StickySessionType': 'insert'}
        self.assert_dict_equals(expected, self.listener.get_optional_params())