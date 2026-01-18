import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import LB_ALB_PARAMS
from libcloud.loadbalancer.base import Member
from libcloud.loadbalancer.types import State
from libcloud.test.file_fixtures import LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.alb import ApplicationLBDriver
class ApplicationLBMockHttp(MockHttp):
    fixtures = LoadBalancerFileFixtures('alb')

    def _2015_12_01_DescribeLoadBalancers(self, method, url, body, headers):
        body = self.fixtures.load('describe_load_balancers.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _2015_12_01_DescribeListeners(self, method, url, body, headers):
        body = self.fixtures.load('describe_load_balancer_listeters.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _2015_12_01_DescribeRules(self, method, url, body, headers):
        body = self.fixtures.load('describe_load_balancer_rules.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _2015_12_01_DescribeTargetGroups(self, method, url, body, headers):
        body = self.fixtures.load('describe_load_balancer_target_groups.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _2015_12_01_DescribeTargetHealth(self, method, url, body, headers):
        body = self.fixtures.load('describe_target_health.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _2015_12_01_DescribeTags(self, method, url, body, headers):
        body = self.fixtures.load('describe_tags.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _2015_12_01_CreateLoadBalancer(self, method, url, body, headers):
        body = self.fixtures.load('create_balancer.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _2015_12_01_CreateTargetGroup(self, method, url, body, headers):
        body = self.fixtures.load('create_target_group.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _2015_12_01_CreateListener(self, method, url, body, headers):
        body = self.fixtures.load('create_listener.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _2015_12_01_CreateRule(self, method, url, body, headers):
        body = self.fixtures.load('create_rule.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _2015_12_01_RegisterTargets(self, method, url, body, headers):
        body = self.fixtures.load('register_targets.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])