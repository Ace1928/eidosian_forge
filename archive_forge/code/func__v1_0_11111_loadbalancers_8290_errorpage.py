import sys
import datetime
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, urlencode
from libcloud.common.types import LibcloudError
from libcloud.loadbalancer.base import Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import MemberCondition
from libcloud.test.file_fixtures import OpenStackFixtures, LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.rackspace import (
def _v1_0_11111_loadbalancers_8290_errorpage(self, method, url, body, headers):
    if method == 'GET':
        body = self.fixtures.load('v1_slug_loadbalancers_8290_errorpage.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])
    elif method == 'PUT':
        json_body = json.loads(body)
        self.assertEqual('<html>Generic Error Page</html>', json_body['errorpage']['content'])
        return (httplib.ACCEPTED, '', {}, httplib.responses[httplib.ACCEPTED])
    raise NotImplementedError