import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import LB_ELB_PARAMS
from libcloud.loadbalancer.base import Member, Algorithm
from libcloud.loadbalancer.types import State
from libcloud.test.file_fixtures import LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.elb import ElasticLBDriver
class ElasticLBMockHttp(MockHttp):
    fixtures = LoadBalancerFileFixtures('elb')

    def _2012_06_01_DescribeLoadBalancers(self, method, url, body, headers):
        body = self.fixtures.load('describe_load_balancers.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _2012_06_01_DescribeTags(self, method, url, body, headers):
        body = self.fixtures.load('describe_tags.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _2012_06_01_CreateLoadBalancer(self, method, url, body, headers):
        body = self.fixtures.load('create_load_balancer.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _2012_06_01_DeregisterInstancesFromLoadBalancer(self, method, url, body, headers):
        body = self.fixtures.load('deregister_instances_from_load_balancer.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _2012_06_01_CreateLoadBalancerPolicy(self, method, url, body, headers):
        body = self.fixtures.load('create_load_balancer_policy.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _2012_06_01_DeleteLoadBalancer(self, method, url, body, headers):
        body = ''
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _2012_06_01_DescribeLoadBalancerPolicies(self, method, url, body, headers):
        body = self.fixtures.load('describe_load_balancer_policies.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _2012_06_01_DescribeLoadBalancerPolicyTypes(self, method, url, body, headers):
        body = self.fixtures.load('describe_load_balancers_policy_types.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _2012_06_01_DeleteLoadBalancerPolicy(self, method, url, body, headers):
        body = ''
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _2012_06_01_SetLoadBalancerPoliciesOfListener(self, method, url, body, headers):
        body = self.fixtures.load('set_load_balancer_policies_of_listener.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _2012_06_01_SetLoadBalancerPoliciesForBackendServer(self, method, url, body, headers):
        body = self.fixtures.load('set_load_balancer_policies_for_backend_server.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])