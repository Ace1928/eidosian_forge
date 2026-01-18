import sys
import datetime
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, urlencode
from libcloud.common.types import LibcloudError
from libcloud.loadbalancer.base import Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import MemberCondition
from libcloud.test.file_fixtures import OpenStackFixtures, LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.rackspace import (
def _v1_0_11111_loadbalancers_8291_nodes(self, method, url, body, headers):
    if method == 'POST':
        json_body = json.loads(body)
        json_node = json_body['nodes'][0]
        self.assertEqual('ENABLED', json_node['condition'])
        response_body = self.fixtures.load('v1_slug_loadbalancers_8290_nodes_post.json')
        return (httplib.ACCEPTED, response_body, {}, httplib.responses[httplib.ACCEPTED])
    raise NotImplementedError