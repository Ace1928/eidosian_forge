import sys
import datetime
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, urlencode
from libcloud.common.types import LibcloudError
from libcloud.loadbalancer.base import Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import MemberCondition
from libcloud.test.file_fixtures import OpenStackFixtures, LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.rackspace import (
class RackspaceLBMockHttp(MockHttp, unittest.TestCase):
    fixtures = LoadBalancerFileFixtures('rackspace')
    auth_fixtures = OpenStackFixtures()

    def _v2_0_tokens(self, method, url, body, headers):
        body = self.fixtures.load('_v2_0__auth.json')
        return (httplib.OK, body, headers, httplib.responses[httplib.OK])

    def _v1_0_11111_loadbalancers_protocols(self, method, url, body, headers):
        body = self.fixtures.load('v1_slug_loadbalancers_protocols.json')
        return (httplib.ACCEPTED, body, {}, httplib.responses[httplib.ACCEPTED])

    def _v1_0_11111_loadbalancers_algorithms(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('v1_slug_loadbalancers_algorithms.json')
            return (httplib.ACCEPTED, body, {}, httplib.responses[httplib.ACCEPTED])
        raise NotImplementedError

    def _v1_0_11111_loadbalancers(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('v1_slug_loadbalancers.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        elif method == 'POST':
            json_body = json.loads(body)
            loadbalancer_json = json_body['loadBalancer']
            member_1_json, member_2_json = loadbalancer_json['nodes']
            self.assertEqual(loadbalancer_json['protocol'], 'HTTP')
            self.assertEqual(loadbalancer_json['algorithm'], 'ROUND_ROBIN')
            self.assertEqual(loadbalancer_json['virtualIps'][0]['type'], 'PUBLIC')
            self.assertEqual(member_1_json['condition'], 'DISABLED')
            self.assertEqual(member_1_json['weight'], 10)
            self.assertEqual(member_2_json['condition'], 'ENABLED')
            body = self.fixtures.load('v1_slug_loadbalancers_post.json')
            return (httplib.ACCEPTED, body, {}, httplib.responses[httplib.ACCEPTED])
        elif method == 'DELETE':
            balancers = self.fixtures.load('v1_slug_loadbalancers.json')
            balancers_json = json.loads(balancers)
            for balancer in balancers_json['loadBalancers']:
                id = balancer['id']
                self.assertTrue(urlencode([('id', id)]) in url, msg='Did not delete balancer with id %d' % id)
            return (httplib.ACCEPTED, '', {}, httplib.responses[httplib.ACCEPTED])
        raise NotImplementedError

    def _v1_0_11111_loadbalancers_EX_MEMBER_ADDRESS(self, method, url, body, headers):
        body = self.fixtures.load('v1_slug_loadbalancers_nodeaddress.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_0_11111_loadbalancers_8155(self, method, url, body, headers):
        if method == 'DELETE':
            return (httplib.ACCEPTED, '', {}, httplib.responses[httplib.ACCEPTED])
        raise NotImplementedError

    def _v1_0_11111_loadbalancers_8290(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('v1_slug_loadbalancers_8290.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        raise NotImplementedError

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

    def _v1_0_11111_loadbalancers_8291(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('v1_slug_loadbalancers_8291.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        raise NotImplementedError

    def _v1_0_11111_loadbalancers_8291_nodes(self, method, url, body, headers):
        if method == 'POST':
            json_body = json.loads(body)
            json_node = json_body['nodes'][0]
            self.assertEqual('ENABLED', json_node['condition'])
            response_body = self.fixtures.load('v1_slug_loadbalancers_8290_nodes_post.json')
            return (httplib.ACCEPTED, response_body, {}, httplib.responses[httplib.ACCEPTED])
        raise NotImplementedError

    def _v1_0_11111_loadbalancers_8292(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('v1_slug_loadbalancers_8292.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        raise NotImplementedError

    def _v1_0_11111_loadbalancers_8292_nodes(self, method, url, body, headers):
        if method == 'POST':
            json_body = json.loads(body)
            json_node_1 = json_body['nodes'][0]
            json_node_2 = json_body['nodes'][1]
            self.assertEqual('10.1.0.12', json_node_1['address'])
            self.assertEqual('10.1.0.13', json_node_2['address'])
            response_body = self.fixtures.load('v1_slug_loadbalancers_8292_nodes_post.json')
            return (httplib.ACCEPTED, response_body, {}, httplib.responses[httplib.ACCEPTED])
        raise NotImplementedError

    def _v1_0_11111_loadbalancers_8290_nodes_30944(self, method, url, body, headers):
        if method == 'PUT':
            json_body = json.loads(body)
            self.assertEqual('ENABLED', json_body['condition'])
            self.assertEqual(12, json_body['weight'])
            return (httplib.ACCEPTED, '', {}, httplib.responses[httplib.ACCEPTED])
        elif method == 'DELETE':
            return (httplib.ACCEPTED, '', {}, httplib.responses[httplib.ACCEPTED])
        raise NotImplementedError

    def _v1_0_11111_loadbalancers_8290_healthmonitor(self, method, url, body, headers):
        if method == 'DELETE':
            return (httplib.ACCEPTED, '', {}, httplib.responses[httplib.ACCEPTED])
        raise NotImplementedError

    def _v1_0_11111_loadbalancers_8290_connectionthrottle(self, method, url, body, headers):
        if method == 'DELETE':
            return (httplib.ACCEPTED, '', {}, httplib.responses[httplib.ACCEPTED])
        raise NotImplementedError

    def _v1_0_11111_loadbalancers_8290_connectionlogging(self, method, url, body, headers):
        if method == 'PUT':
            json_body = json.loads(body)
            self.assertFalse(json_body['connectionLogging']['enabled'])
            return (httplib.ACCEPTED, '', {}, httplib.responses[httplib.ACCEPTED])
        raise NotImplementedError

    def _v1_0_11111_loadbalancers_8290_sessionpersistence(self, method, url, body, headers):
        if method == 'DELETE':
            return (httplib.ACCEPTED, '', {}, httplib.responses[httplib.ACCEPTED])
        raise NotImplementedError

    def _v1_0_11111_loadbalancers_8290_errorpage(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('v1_slug_loadbalancers_8290_errorpage.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        elif method == 'PUT':
            json_body = json.loads(body)
            self.assertEqual('<html>Generic Error Page</html>', json_body['errorpage']['content'])
            return (httplib.ACCEPTED, '', {}, httplib.responses[httplib.ACCEPTED])
        raise NotImplementedError

    def _v1_0_11111_loadbalancers_18940(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('v1_slug_loadbalancers_18940_ex_public_ips.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        raise NotImplementedError

    def _v1_0_11111_loadbalancers_18945(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('v1_slug_loadbalancers_18945_ex_public_ips.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        raise NotImplementedError

    def _v1_0_11111_loadbalancers_18940_errorpage(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('v1_slug_loadbalancers_18940_errorpage.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        raise NotImplementedError

    def _v1_0_11111_loadbalancers_18940_accesslist(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('v1_slug_loadbalancers_18940_accesslist.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        raise NotImplementedError

    def _v1_0_11111_loadbalancers_18941(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('v1_slug_loadbalancers_18941_ex_private_ips.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        raise NotImplementedError

    def _v1_0_11111_loadbalancers_94692(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('v1_slug_loadbalancers_94692_weighted_round_robin.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        raise NotImplementedError

    def _v1_0_11111_loadbalancers_94693(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('v1_slug_loadbalancers_94693_weighted_least_connections.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        raise NotImplementedError

    def _v1_0_11111_loadbalancers_94694(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('v1_slug_loadbalancers_94694_unknown_algorithm.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        raise NotImplementedError

    def _v1_0_11111_loadbalancers_94695(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('v1_slug_loadbalancers_94695_full_details.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        raise NotImplementedError

    def _v1_0_11111_loadbalancers_94695_healthmonitor(self, method, url, body, headers):
        if method == 'PUT':
            json_body = json.loads(body)
            self.assertEqual('CONNECT', json_body['type'])
            self.assertEqual(10, json_body['delay'])
            self.assertEqual(5, json_body['timeout'])
            self.assertEqual(2, json_body['attemptsBeforeDeactivation'])
            return (httplib.ACCEPTED, '', {}, httplib.responses[httplib.ACCEPTED])
        raise NotImplementedError

    def _v1_0_11111_loadbalancers_94695_connectionthrottle(self, method, url, body, headers):
        if method == 'PUT':
            json_body = json.loads(body)
            self.assertEqual(50, json_body['minConnections'])
            self.assertEqual(200, json_body['maxConnections'])
            self.assertEqual(50, json_body['maxConnectionRate'])
            self.assertEqual(10, json_body['rateInterval'])
            return (httplib.ACCEPTED, '', {}, httplib.responses[httplib.ACCEPTED])
        raise NotImplementedError

    def _v1_0_11111_loadbalancers_94695_connectionlogging(self, method, url, body, headers):
        if method == 'PUT':
            json_body = json.loads(body)
            self.assertTrue(json_body['connectionLogging']['enabled'])
            return (httplib.ACCEPTED, '', {}, httplib.responses[httplib.ACCEPTED])
        raise NotImplementedError

    def _v1_0_11111_loadbalancers_94695_sessionpersistence(self, method, url, body, headers):
        if method == 'PUT':
            json_body = json.loads(body)
            persistence_type = json_body['sessionPersistence']['persistenceType']
            self.assertEqual('HTTP_COOKIE', persistence_type)
            return (httplib.ACCEPTED, '', {}, httplib.responses[httplib.ACCEPTED])
        raise NotImplementedError

    def _v1_0_11111_loadbalancers_94695_errorpage(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('error_page_default.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        elif method == 'DELETE':
            return (httplib.OK, '', {}, httplib.responses[httplib.OK])
        raise NotImplementedError

    def _v1_0_11111_loadbalancers_94696(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('v1_slug_loadbalancers_94696_http_health_monitor.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        raise NotImplementedError

    def _v1_0_11111_loadbalancers_94696_healthmonitor(self, method, url, body, headers):
        if method == 'PUT':
            json_body = json.loads(body)
            self.assertEqual('HTTP', json_body['type'])
            self.assertEqual(10, json_body['delay'])
            self.assertEqual(5, json_body['timeout'])
            self.assertEqual(2, json_body['attemptsBeforeDeactivation'])
            self.assertEqual('/', json_body['path'])
            self.assertEqual('^[234][0-9][0-9]$', json_body['statusRegex'])
            self.assertEqual('Hello World!', json_body['bodyRegex'])
            return (httplib.ACCEPTED, '', {}, httplib.responses[httplib.ACCEPTED])
        raise NotImplementedError

    def _v1_0_11111_loadbalancers_94697(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('v1_slug_loadbalancers_94697_https_health_monitor.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        raise NotImplementedError

    def _v1_0_11111_loadbalancers_94698(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('v1_slug_loadbalancers_94698_with_access_list.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        raise NotImplementedError

    def _v1_0_11111_loadbalancers_94698_accesslist(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('v1_slug_loadbalancers_94698_accesslist.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        elif method == 'POST':
            json_body = json.loads(body)
            self.assertEqual('0.0.0.0/0', json_body['networkItem']['address'])
            self.assertEqual('DENY', json_body['networkItem']['type'])
            return (httplib.ACCEPTED, '', {}, httplib.responses[httplib.ACCEPTED])
        raise NotImplementedError

    def _v1_0_11111_loadbalancers_94699(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('v1_slug_loadbalancers_94698_with_access_list.json')
            json_body = json.loads(body)
            json_body['loadBalancer']['id'] = 94699
            updated_body = json.dumps(json_body)
            return (httplib.OK, updated_body, {}, httplib.responses[httplib.OK])
        raise NotImplementedError

    def _v1_0_11111_loadbalancers_94699_accesslist(self, method, url, body, headers):
        if method == 'DELETE':
            fixture = 'v1_slug_loadbalancers_94698_with_access_list.json'
            fixture_json = json.loads(self.fixtures.load(fixture))
            access_list_json = fixture_json['loadBalancer']['accessList']
            for access_rule in access_list_json:
                id = access_rule['id']
                self.assertTrue(urlencode([('id', id)]) in url, msg='Did not delete access rule with id %d' % id)
            return (httplib.ACCEPTED, '', {}, httplib.responses[httplib.ACCEPTED])
        elif method == 'POST':
            json_body = json.loads(body)
            access_list = json_body['accessList']
            self.assertEqual('ALLOW', access_list[0]['type'])
            self.assertEqual('2001:4801:7901::6/64', access_list[0]['address'])
            self.assertEqual('DENY', access_list[1]['type'])
            self.assertEqual('8.8.8.8/0', access_list[1]['address'])
            return (httplib.ACCEPTED, '', {}, httplib.responses[httplib.ACCEPTED])
        raise NotImplementedError

    def _v1_0_11111_loadbalancers_94698_accesslist_1007(self, method, url, body, headers):
        if method == 'DELETE':
            return (httplib.ACCEPTED, '', {}, httplib.responses[httplib.ACCEPTED])
        raise NotImplementedError

    def _v1_0_11111_loadbalancers_94700(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('v1_slug_loadbalancers_94700_http_health_monitor_no_body_regex.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        raise NotImplementedError

    def _v1_0_11111_loadbalancers_94700_healthmonitor(self, method, url, body, headers):
        if method == 'PUT':
            json_body = json.loads(body)
            self.assertEqual('HTTP', json_body['type'])
            self.assertEqual(10, json_body['delay'])
            self.assertEqual(5, json_body['timeout'])
            self.assertEqual(2, json_body['attemptsBeforeDeactivation'])
            self.assertEqual('/', json_body['path'])
            self.assertEqual('^[234][0-9][0-9]$', json_body['statusRegex'])
            self.assertFalse('bodyRegex' in json_body)
            return (httplib.ACCEPTED, '', {}, httplib.responses[httplib.ACCEPTED])
        raise NotImplementedError

    def _v1_0_11111_loadbalancers_3130(self, method, url, body, headers):
        """update_balancer(b, protocol='HTTPS'), then get_balancer('3130')"""
        if method == 'PUT':
            json_body = json.loads(body)
            self.assertDictEqual(json_body, {'protocol': 'HTTPS'})
            return (httplib.ACCEPTED, '', {}, httplib.responses[httplib.ACCEPTED])
        elif method == 'GET':
            response_body = json.loads(self.fixtures.load('v1_slug_loadbalancers_3xxx.json'))
            response_body['loadBalancer']['id'] = 3130
            response_body['loadBalancer']['protocol'] = 'HTTPS'
            return (httplib.OK, json.dumps(response_body), {}, httplib.responses[httplib.OK])
        raise NotImplementedError

    def _v1_0_11111_loadbalancers_3131(self, method, url, body, headers):
        """update_balancer(b, port=443), then get_balancer('3131')"""
        if method == 'PUT':
            json_body = json.loads(body)
            self.assertDictEqual(json_body, {'port': 1337})
            return (httplib.ACCEPTED, '', {}, httplib.responses[httplib.ACCEPTED])
        elif method == 'GET':
            response_body = json.loads(self.fixtures.load('v1_slug_loadbalancers_3xxx.json'))
            response_body['loadBalancer']['id'] = 3131
            response_body['loadBalancer']['port'] = 1337
            return (httplib.OK, json.dumps(response_body), {}, httplib.responses[httplib.OK])
        raise NotImplementedError

    def _v1_0_11111_loadbalancers_3132(self, method, url, body, headers):
        """update_balancer(b, name='new_lb_name'), then get_balancer('3132')"""
        if method == 'PUT':
            json_body = json.loads(body)
            self.assertDictEqual(json_body, {'name': 'new_lb_name'})
            return (httplib.ACCEPTED, '', {}, httplib.responses[httplib.ACCEPTED])
        elif method == 'GET':
            response_body = json.loads(self.fixtures.load('v1_slug_loadbalancers_3xxx.json'))
            response_body['loadBalancer']['id'] = 3132
            response_body['loadBalancer']['name'] = 'new_lb_name'
            return (httplib.OK, json.dumps(response_body), {}, httplib.responses[httplib.OK])
        raise NotImplementedError

    def _v1_0_11111_loadbalancers_3133(self, method, url, body, headers):
        """update_balancer(b, algorithm='ROUND_ROBIN'), then get_balancer('3133')"""
        if method == 'PUT':
            json_body = json.loads(body)
            self.assertDictEqual(json_body, {'algorithm': 'ROUND_ROBIN'})
            return (httplib.ACCEPTED, '', {}, httplib.responses[httplib.ACCEPTED])
        elif method == 'GET':
            response_body = json.loads(self.fixtures.load('v1_slug_loadbalancers_3xxx.json'))
            response_body['loadBalancer']['id'] = 3133
            response_body['loadBalancer']['algorithm'] = 'ROUND_ROBIN'
            return (httplib.OK, json.dumps(response_body), {}, httplib.responses[httplib.OK])
        raise NotImplementedError

    def _v1_0_11111_loadbalancers_3134(self, method, url, body, headers):
        """update.balancer(b, algorithm='HAVE_MERCY_ON_OUR_SERVERS')"""
        if method == 'PUT':
            return (httplib.BAD_REQUEST, '', {}, httplib.responses[httplib.BAD_REQUEST])
        raise NotImplementedError

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

    def _v1_0_11111_loadbalancers_3136(self, method, url, body, headers):
        """update_balancer(b, protocol='IMAPv3'), then get_balancer('3136')"""
        if method == 'PUT':
            json_body = json.loads(body)
            self.assertDictEqual(json_body, {'protocol': 'IMAPv3'})
            return (httplib.ACCEPTED, '', {}, httplib.responses[httplib.ACCEPTED])
        elif method == 'GET':
            response_body = json.loads(self.fixtures.load('v1_slug_loadbalancers_3xxx.json'))
            response_body['loadBalancer']['id'] = 3136
            response_body['loadBalancer']['protocol'] = 'IMAPv3'
            return (httplib.OK, json.dumps(response_body), {}, httplib.responses[httplib.OK])
        raise NotImplementedError

    def _v1_0_11111_loadbalancers_3137(self, method, url, body, headers):
        """update_balancer(b, protocol='IMAPv3'), then get_balancer('3137')"""
        if method == 'PUT':
            json_body = json.loads(body)
            self.assertDictEqual(json_body, {'protocol': 'IMAPv4'})
            return (httplib.ACCEPTED, '', {}, httplib.responses[httplib.ACCEPTED])
        elif method == 'GET':
            response_body = json.loads(self.fixtures.load('v1_slug_loadbalancers_3xxx.json'))
            response_body['loadBalancer']['id'] = 3137
            response_body['loadBalancer']['protocol'] = 'IMAPv4'
            return (httplib.OK, json.dumps(response_body), {}, httplib.responses[httplib.OK])
        raise NotImplementedError

    def _v1_0_11111_loadbalancers_8290_usage_current(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('v1_0_slug_loadbalancers_8290_usage_current.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        raise NotImplementedError