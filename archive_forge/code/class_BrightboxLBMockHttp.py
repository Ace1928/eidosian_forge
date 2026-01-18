import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import LB_BRIGHTBOX_PARAMS
from libcloud.loadbalancer.base import Member, Algorithm
from libcloud.loadbalancer.types import State
from libcloud.test.file_fixtures import LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.brightbox import BrightboxLBDriver
class BrightboxLBMockHttp(MockHttp):
    fixtures = LoadBalancerFileFixtures('brightbox')

    def _token(self, method, url, body, headers):
        if method == 'POST':
            return (httplib.OK, self.fixtures.load('token.json'), {'content-type': 'application/json'}, httplib.responses[httplib.OK])

    def _1_0_load_balancers(self, method, url, body, headers):
        if method == 'GET':
            return (httplib.OK, self.fixtures.load('load_balancers.json'), {'content-type': 'application/json'}, httplib.responses[httplib.OK])
        elif method == 'POST':
            body = self.fixtures.load('load_balancers_post.json')
            return (httplib.ACCEPTED, body, {'content-type': 'application/json'}, httplib.responses[httplib.ACCEPTED])

    def _1_0_load_balancers_lba_1235f(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('load_balancers_lba_1235f.json')
            return (httplib.OK, body, {'content-type': 'application/json'}, httplib.responses[httplib.OK])
        elif method == 'DELETE':
            return (httplib.ACCEPTED, '', {'content-type': 'application/json'}, httplib.responses[httplib.ACCEPTED])

    def _1_0_load_balancers_lba_1235f_add_nodes(self, method, url, body, headers):
        if method == 'POST':
            return (httplib.ACCEPTED, '', {'content-type': 'application/json'}, httplib.responses[httplib.ACCEPTED])

    def _1_0_load_balancers_lba_1235f_remove_nodes(self, method, url, body, headers):
        if method == 'POST':
            return (httplib.ACCEPTED, '', {'content-type': 'application/json'}, httplib.responses[httplib.ACCEPTED])