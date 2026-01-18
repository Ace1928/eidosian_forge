import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import LB_ELB_PARAMS
from libcloud.loadbalancer.base import Member, Algorithm
from libcloud.loadbalancer.types import State
from libcloud.test.file_fixtures import LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.elb import ElasticLBDriver
class ElasticLBTests(unittest.TestCase):

    def setUp(self):
        ElasticLBMockHttp.test = self
        ElasticLBDriver.connectionCls.conn_class = ElasticLBMockHttp
        ElasticLBMockHttp.type = None
        ElasticLBMockHttp.use_param = 'Action'
        self.driver = ElasticLBDriver(*LB_ELB_PARAMS)

    def test_instantiate_driver_with_token(self):
        token = 'temporary_credentials_token'
        driver = ElasticLBDriver(*LB_ELB_PARAMS, **{'token': token})
        self.assertTrue(hasattr(driver, 'token'), 'Driver has no attribute token')
        self.assertEqual(token, driver.token, 'Driver token does not match with provided token')

    def test_driver_with_token_signature_version(self):
        token = 'temporary_credentials_token'
        driver = ElasticLBDriver(*LB_ELB_PARAMS, **{'token': token})
        kwargs = driver._ex_connection_class_kwargs()
        self.assertTrue('signature_version' in kwargs, 'Driver has no attribute signature_version')
        self.assertEqual('4', kwargs['signature_version'], 'Signature version is not 4 with temporary credentials')

    def test_list_protocols(self):
        protocols = self.driver.list_protocols()
        self.assertEqual(len(protocols), 4)
        self.assertTrue('tcp' in protocols)
        self.assertTrue('http' in protocols)

    def test_list_balancers(self):
        balancers = self.driver.list_balancers()
        self.assertEqual(len(balancers), 1)
        self.assertEqual(balancers[0].id, 'tests')
        self.assertEqual(balancers[0].name, 'tests')

    def test_list_balancers_with_tags(self):
        balancers = self.driver.list_balancers(ex_fetch_tags=True)
        self.assertEqual(len(balancers), 1)
        self.assertEqual(balancers[0].id, 'tests')
        self.assertEqual(balancers[0].name, 'tests')
        self.assertTrue('tags' in balancers[0].extra, 'No tags dict found in balancer.extra')
        self.assertEqual(balancers[0].extra['tags']['project'], 'lima')

    def test_list_balancer_tags(self):
        tags = self.driver._ex_list_balancer_tags('tests')
        self.assertEqual(len(tags), 1)
        self.assertEqual(tags['project'], 'lima')

    def test_get_balancer(self):
        balancer = self.driver.get_balancer(balancer_id='tests')
        self.assertEqual(balancer.id, 'tests')
        self.assertEqual(balancer.name, 'tests')
        self.assertEqual(balancer.state, State.UNKNOWN)

    def test_get_balancer_with_tags(self):
        balancer = self.driver.get_balancer(balancer_id='tests', ex_fetch_tags=True)
        self.assertEqual(balancer.id, 'tests')
        self.assertEqual(balancer.name, 'tests')
        self.assertTrue('tags' in balancer.extra, 'No tags dict found in balancer.extra')
        self.assertEqual(balancer.extra['tags']['project'], 'lima')

    def test_populate_balancer_tags(self):
        balancer = self.driver.get_balancer(balancer_id='tests')
        balancer = self.driver._ex_populate_balancer_tags(balancer)
        self.assertEqual(balancer.id, 'tests')
        self.assertEqual(balancer.name, 'tests')
        self.assertTrue('tags' in balancer.extra, 'No tags dict found in balancer.extra')
        self.assertEqual(balancer.extra['tags']['project'], 'lima')

    def test_destroy_balancer(self):
        balancer = self.driver.get_balancer(balancer_id='tests')
        self.assertTrue(self.driver.destroy_balancer(balancer))

    def test_create_balancer(self):
        members = [Member('srv-lv426', None, None)]
        balancer = self.driver.create_balancer(name='lb2', port=80, protocol='http', algorithm=Algorithm.ROUND_ROBIN, members=members)
        self.assertEqual(balancer.name, 'lb2')
        self.assertEqual(balancer.port, 80)
        self.assertEqual(balancer.state, State.PENDING)

    def test_balancer_list_members(self):
        balancer = self.driver.get_balancer(balancer_id='tests')
        members = balancer.list_members()
        self.assertEqual(len(members), 1)
        self.assertEqual(members[0].balancer, balancer)
        self.assertEqual('i-64bd081c', members[0].id)

    def test_balancer_detach_member(self):
        balancer = self.driver.get_balancer(balancer_id='lba-1235f')
        member = Member('i-64bd081c', None, None)
        self.assertTrue(balancer.detach_member(member))

    def test_ex_list_balancer_policies(self):
        balancer = self.driver.get_balancer(balancer_id='tests')
        policies = self.driver.ex_list_balancer_policies(balancer)
        self.assertTrue('MyDurationStickyPolicy' in policies)

    def test_ex_list_balancer_policy_types(self):
        policy_types = self.driver.ex_list_balancer_policy_types()
        self.assertTrue('ProxyProtocolPolicyType' in policy_types)

    def test_ex_create_balancer_policy(self):
        self.assertTrue(self.driver.ex_create_balancer_policy(name='tests', policy_name='MyDurationProxyPolicy', policy_type='ProxyProtocolPolicyType'))

    def test_ex_delete_balancer_policy(self):
        self.assertTrue(self.driver.ex_delete_balancer_policy(name='tests', policy_name='MyDurationProxyPolicy'))

    def test_ex_set_balancer_policies_listener(self):
        self.assertTrue(self.driver.ex_set_balancer_policies_listener(name='tests', port=80, policies=['MyDurationStickyPolicy']))

    def test_ex_set_balancer_policies_backend_server(self):
        self.assertTrue(self.driver.ex_set_balancer_policies_backend_server(name='tests', instance_port=80, policies=['MyDurationProxyPolicy']))

    def text_ex_create_balancer_listeners(self):
        self.assertTrue(self.driver.ex_create_balancer_listeners(name='tests', listeners=[[1024, 65533, 'HTTP']]))