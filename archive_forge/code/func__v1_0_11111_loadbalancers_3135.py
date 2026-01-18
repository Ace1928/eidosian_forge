import sys
import datetime
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, urlencode
from libcloud.common.types import LibcloudError
from libcloud.loadbalancer.base import Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import MemberCondition
from libcloud.test.file_fixtures import OpenStackFixtures, LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.rackspace import (
def _v1_0_11111_loadbalancers_3135(self, method, url, body, headers):
    """update_balancer(b, protocol='IMAPv3'), then get_balancer('3135')"""
    if method == 'PUT':
        json_body = json.loads(body)
        self.assertDictEqual(json_body, {'protocol': 'IMAPv2'})
        return (httplib.ACCEPTED, '', {}, httplib.responses[httplib.ACCEPTED])
    elif method == 'GET':
        response_body = json.loads(self.fixtures.load('v1_slug_loadbalancers_3xxx.json'))
        response_body['loadBalancer']['id'] = 3135
        response_body['loadBalancer']['protocol'] = 'IMAPv2'
        return (httplib.OK, json.dumps(response_body), {}, httplib.responses[httplib.OK])
    raise NotImplementedError