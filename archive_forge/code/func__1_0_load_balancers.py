import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import LB_BRIGHTBOX_PARAMS
from libcloud.loadbalancer.base import Member, Algorithm
from libcloud.loadbalancer.types import State
from libcloud.test.file_fixtures import LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.brightbox import BrightboxLBDriver
def _1_0_load_balancers(self, method, url, body, headers):
    if method == 'GET':
        return (httplib.OK, self.fixtures.load('load_balancers.json'), {'content-type': 'application/json'}, httplib.responses[httplib.OK])
    elif method == 'POST':
        body = self.fixtures.load('load_balancers_post.json')
        return (httplib.ACCEPTED, body, {'content-type': 'application/json'}, httplib.responses[httplib.ACCEPTED])