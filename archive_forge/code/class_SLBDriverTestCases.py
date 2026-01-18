import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node
from libcloud.test.secrets import LB_SLB_PARAMS
from libcloud.compute.types import NodeState
from libcloud.loadbalancer.base import Member, Algorithm
from libcloud.loadbalancer.types import State
from libcloud.test.file_fixtures import LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.slb import (
class SLBDriverTestCases(unittest.TestCase):
    region = LB_SLB_PARAMS[2]

    def setUp(self):
        SLBMockHttp.test = self
        SLBDriver.connectionCls.conn_class = SLBMockHttp
        SLBMockHttp.type = None
        SLBMockHttp.use_param = 'Action'
        self.driver = SLBDriver(*LB_SLB_PARAMS)

    def test_list_protocols(self):
        protocols = self.driver.list_protocols()
        self.assertEqual(4, len(protocols))
        expected = ['tcp', 'udp', 'http', 'https']
        diff = set(expected) - set(protocols)
        self.assertEqual(0, len(diff))

    def test_list_balancers(self):
        balancers = self.driver.list_balancers()
        self.assertEqual(len(balancers), 1)
        balancer = balancers[0]
        self.assertEqual('15229f88562-cn-hangzhou-dg-a01', balancer.id)
        self.assertEqual('abc', balancer.name)
        self.assertEqual(State.RUNNING, balancer.state)
        self.assertEqual('120.27.186.149', balancer.ip)
        self.assertTrue(balancer.port is None)
        self.assertEqual(self.driver, balancer.driver)
        expected_extra = {'create_timestamp': 1452403099000, 'address_type': 'internet', 'region_id': 'cn-hangzhou-dg-a01', 'region_id_alias': 'cn-hangzhou', 'create_time': '2016-01-10T13:18Z', 'master_zone_id': 'cn-hangzhou-d', 'slave_zone_id': 'cn-hangzhou-b', 'network_type': 'classic'}
        self._validate_extras(expected_extra, balancer.extra)

    def _validate_extras(self, expected, actual):
        self.assertTrue(actual is not None)
        for key, value in iter(expected.items()):
            self.assertTrue(key in actual)
            self.assertEqual(value, actual[key], 'extra %(key)s not equal, expected: "%(expected)s", actual: "%(actual)s"' % {'key': key, 'expected': value, 'actual': actual[key]})

    def test_list_balancers_with_ids(self):
        SLBMockHttp.type = 'list_balancers_ids'
        self.balancer_ids = ['id1', 'id2']
        balancers = self.driver.list_balancers(ex_balancer_ids=self.balancer_ids)
        self.assertTrue(balancers is not None)

    def test_list_balancers_with_ex_filters(self):
        SLBMockHttp.type = 'list_balancers_filters'
        self.ex_filters = {'AddressType': 'internet'}
        balancers = self.driver.list_balancers(ex_filters=self.ex_filters)
        self.assertTrue(balancers is not None)

    def test_get_balancer(self):
        SLBMockHttp.type = 'get_balancer'
        balancer = self.driver.get_balancer(balancer_id='tests')
        self.assertEqual(balancer.id, '15229f88562-cn-hangzhou-dg-a01')
        self.assertEqual(balancer.name, 'abc')
        self.assertEqual(balancer.state, State.RUNNING)

    def test_destroy_balancer(self):
        balancer = self.driver.get_balancer(balancer_id='tests')
        self.assertTrue(self.driver.destroy_balancer(balancer))

    def test_create_balancer(self):
        self.name = 'balancer1'
        self.port = 80
        self.protocol = 'http'
        self.algorithm = Algorithm.WEIGHTED_ROUND_ROBIN
        self.extra = {'ex_address_type': 'internet', 'ex_internet_charge_type': 'paybytraffic', 'ex_bandwidth': 1, 'ex_master_zone_id': 'cn-hangzhou-d', 'ex_slave_zone_id': 'cn-hangzhou-b', 'StickySession': 'on', 'HealthCheck': 'on'}
        self.members = [Member('node1', None, None)]
        balancer = self.driver.create_balancer(name=self.name, port=self.port, protocol=self.protocol, algorithm=self.algorithm, members=self.members, **self.extra)
        self.assertEqual(balancer.name, self.name)
        self.assertEqual(balancer.port, self.port)
        self.assertEqual(balancer.state, State.UNKNOWN)

    def test_create_balancer_no_port_exception(self):
        self.assertRaises(AttributeError, self.driver.create_balancer, None, None, 'http', Algorithm.WEIGHTED_ROUND_ROBIN, None)

    def test_create_balancer_unsupport_protocol_exception(self):
        self.assertRaises(AttributeError, self.driver.create_balancer, None, 443, 'ssl', Algorithm.WEIGHTED_ROUND_ROBIN, None)

    def test_create_balancer_multiple_member_ports_exception(self):
        members = [Member('m1', '1.2.3.4', 80), Member('m2', '1.2.3.5', 81)]
        self.assertRaises(AttributeError, self.driver.create_balancer, None, 80, 'http', Algorithm.WEIGHTED_ROUND_ROBIN, members)

    def test_create_balancer_bandwidth_value_error(self):
        self.assertRaises(AttributeError, self.driver.create_balancer, None, 80, 'http', Algorithm.WEIGHTED_ROUND_ROBIN, None, ex_bandwidth='NAN')

    def test_create_balancer_paybybandwidth_without_bandwidth_exception(self):
        self.assertRaises(AttributeError, self.driver.create_balancer, None, 80, 'http', Algorithm.WEIGHTED_ROUND_ROBIN, None, ex_internet_charge_type='paybybandwidth')

    def test_balancer_list_members(self):
        balancer = self.driver.get_balancer(balancer_id='tests')
        members = balancer.list_members()
        self.assertEqual(len(members), 1)
        self.assertEqual(members[0].balancer, balancer)
        self.assertEqual('i-23tshnsdq', members[0].id)

    def test_balancer_list_listeners(self):
        balancer = self.driver.get_balancer(balancer_id='tests')
        listeners = self.driver.ex_list_listeners(balancer)
        self.assertEqual(1, len(listeners))
        listener = listeners[0]
        self.assertEqual('80', listener.port)

    def test_balancer_detach_member(self):
        self.balancer = self.driver.get_balancer(balancer_id='tests')
        self.member = Member('i-23tshnsdq', None, None)
        self.assertTrue(self.balancer.detach_member(self.member))

    def test_balancer_attach_compute_node(self):
        SLBMockHttp.type = 'attach_compute_node'
        self.balancer = self.driver.get_balancer(balancer_id='tests')
        self.node = Node(id='node1', name='node-name', state=NodeState.RUNNING, public_ips=['1.2.3.4'], private_ips=['4.3.2.1'], driver=self.driver)
        member = self.driver.balancer_attach_compute_node(self.balancer, self.node)
        self.assertEqual(self.node.id, member.id)
        self.assertEqual(self.node.public_ips[0], member.ip)
        self.assertEqual(self.balancer.port, member.port)

    def test_ex_create_listener(self):
        SLBMockHttp.type = 'create_listener'
        self.balancer = self.driver.get_balancer(balancer_id='tests')
        self.backend_port = 80
        self.protocol = 'http'
        self.algorithm = Algorithm.WEIGHTED_ROUND_ROBIN
        self.bandwidth = 1
        self.extra = {'StickySession': 'off', 'HealthCheck': 'off'}
        self.assertTrue(self.driver.ex_create_listener(self.balancer, self.backend_port, self.protocol, self.algorithm, self.bandwidth, **self.extra))

    def test_ex_create_listener_override_port(self):
        SLBMockHttp.type = 'create_listener_override_port'
        self.balancer = self.driver.get_balancer(balancer_id='tests')
        self.backend_port = 80
        self.protocol = 'http'
        self.algorithm = Algorithm.WEIGHTED_ROUND_ROBIN
        self.bandwidth = 1
        self.extra = {'StickySession': 'off', 'HealthCheck': 'off', 'ListenerPort': 8080}
        self.assertTrue(self.driver.ex_create_listener(self.balancer, self.backend_port, self.protocol, self.algorithm, self.bandwidth, **self.extra))

    def test_ex_start_listener(self):
        SLBMockHttp.type = 'start_listener'
        balancer = self.driver.get_balancer(balancer_id='tests')
        self.port = 80
        self.assertTrue(self.driver.ex_start_listener(balancer, self.port))

    def test_ex_stop_listener(self):
        SLBMockHttp.type = 'stop_listener'
        balancer = self.driver.get_balancer(balancer_id='tests')
        self.port = 80
        self.assertTrue(self.driver.ex_stop_listener(balancer, self.port))

    def test_ex_upload_certificate(self):
        self.name = 'cert1'
        self.cert = 'cert-data'
        self.key = 'key-data'
        certificate = self.driver.ex_upload_certificate(self.name, self.cert, self.key)
        self.assertEqual(self.name, certificate.name)
        self.assertEqual('01:DF:AB:CD', certificate.fingerprint)

    def test_ex_list_certificates(self):
        certs = self.driver.ex_list_certificates()
        self.assertEqual(2, len(certs))
        cert = certs[0]
        self.assertEqual('139a00604ad-cn-east-hangzhou-01', cert.id)
        self.assertEqual('abe', cert.name)
        self.assertEqual('A:B:E', cert.fingerprint)

    def test_ex_list_certificates_ids(self):
        SLBMockHttp.type = 'list_certificates_ids'
        self.cert_ids = ['cert1', 'cert2']
        certs = self.driver.ex_list_certificates(certificate_ids=self.cert_ids)
        self.assertTrue(certs is not None)

    def test_ex_delete_certificate(self):
        self.cert_id = 'cert1'
        self.assertTrue(self.driver.ex_delete_certificate(self.cert_id))

    def test_ex_set_certificate_name(self):
        self.cert_id = 'cert1'
        self.cert_name = 'cert-name'
        self.assertTrue(self.driver.ex_set_certificate_name(self.cert_id, self.cert_name))