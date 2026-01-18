import sys
import datetime
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, urlencode
from libcloud.common.types import LibcloudError
from libcloud.loadbalancer.base import Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import MemberCondition
from libcloud.test.file_fixtures import OpenStackFixtures, LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.rackspace import (
def _v1_0_11111_loadbalancers_8290_nodes(self, method, url, body, headers):
    if method == 'GET':
        body = self.fixtures.load('v1_slug_loadbalancers_8290_nodes.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])
    elif method == 'POST':
        json_body = json.loads(body)
        json_node = json_body['nodes'][0]
        self.assertEqual('DISABLED', json_node['condition'])
        self.assertEqual(10, json_node['weight'])
        response_body = self.fixtures.load('v1_slug_loadbalancers_8290_nodes_post.json')
        return (httplib.ACCEPTED, response_body, {}, httplib.responses[httplib.ACCEPTED])
    elif method == 'DELETE':
        nodes = self.fixtures.load('v1_slug_loadbalancers_8290_nodes.json')
        json_nodes = json.loads(nodes)
        for node in json_nodes['nodes']:
            id = node['id']
            self.assertTrue(urlencode([('id', id)]) in url, msg='Did not delete member with id %d' % id)
        return (httplib.ACCEPTED, '', {}, httplib.responses[httplib.ACCEPTED])
    raise NotImplementedError