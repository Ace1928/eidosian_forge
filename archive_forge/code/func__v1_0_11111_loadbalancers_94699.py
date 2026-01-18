import sys
import datetime
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, urlencode
from libcloud.common.types import LibcloudError
from libcloud.loadbalancer.base import Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import MemberCondition
from libcloud.test.file_fixtures import OpenStackFixtures, LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.rackspace import (
def _v1_0_11111_loadbalancers_94699(self, method, url, body, headers):
    if method == 'GET':
        body = self.fixtures.load('v1_slug_loadbalancers_94698_with_access_list.json')
        json_body = json.loads(body)
        json_body['loadBalancer']['id'] = 94699
        updated_body = json.dumps(json_body)
        return (httplib.OK, updated_body, {}, httplib.responses[httplib.OK])
    raise NotImplementedError