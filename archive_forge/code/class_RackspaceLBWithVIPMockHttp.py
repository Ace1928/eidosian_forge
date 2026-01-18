import sys
import datetime
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, urlencode
from libcloud.common.types import LibcloudError
from libcloud.loadbalancer.base import Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import MemberCondition
from libcloud.test.file_fixtures import OpenStackFixtures, LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.rackspace import (
class RackspaceLBWithVIPMockHttp(MockHttp, unittest.TestCase):
    fixtures = LoadBalancerFileFixtures('rackspace')
    auth_fixtures = OpenStackFixtures()

    def _v2_0_tokens(self, method, url, body, headers):
        body = self.fixtures.load('_v2_0__auth.json')
        return (httplib.OK, body, headers, httplib.responses[httplib.OK])

    def _v1_0_11111_loadbalancers(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('v1_slug_loadbalancers.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        elif method == 'POST':
            json_body = json.loads(body)
            loadbalancer_json = json_body['loadBalancer']
            self.assertEqual(loadbalancer_json['virtualIps'][0]['id'], '12af')
            body = self.fixtures.load('v1_slug_loadbalancers_post.json')
            return (httplib.ACCEPTED, body, {}, httplib.responses[httplib.ACCEPTED])
        raise NotImplementedError