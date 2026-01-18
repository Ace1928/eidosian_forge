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
class SLBMockHttp(MockHttp, unittest.TestCase):
    fixtures = LoadBalancerFileFixtures('slb')

    def _DescribeLoadBalancers(self, method, url, body, headers):
        body = self.fixtures.load('describe_load_balancers.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _list_balancers_ids_DescribeLoadBalancers(self, method, url, body, headers):
        params = {'LoadBalancerId': ','.join(self.test.balancer_ids)}
        self.assertUrlContainsQueryParams(url, params)
        body = self.fixtures.load('describe_load_balancers.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _list_balancers_filters_DescribeLoadBalancers(self, method, url, body, headers):
        params = {'AddressType': 'internet'}
        self.assertUrlContainsQueryParams(url, params)
        body = self.fixtures.load('describe_load_balancers.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _get_balancer_DescribeLoadBalancers(self, method, url, body, headers):
        params = {'LoadBalancerId': 'tests'}
        self.assertUrlContainsQueryParams(url, params)
        return self._DescribeLoadBalancers(method, url, body, headers)

    def _DeleteLoadBalancer(self, method, url, body, headers):
        params = {'LoadBalancerId': '15229f88562-cn-hangzhou-dg-a01'}
        self.assertUrlContainsQueryParams(url, params)
        body = self.fixtures.load('delete_load_balancer.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _DescribeLoadBalancerAttribute(self, method, url, body, headers):
        body = self.fixtures.load('describe_load_balancer_attribute.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _CreateLoadBalancer(self, method, url, body, headers):
        params = {'RegionId': self.test.region, 'LoadBalancerName': self.test.name}
        balancer_keys = {'AddressType': 'ex_address_type', 'InternetChargeType': 'ex_internet_charge_type', 'Bandwidth': 'ex_bandwidth', 'MasterZoneId': 'ex_master_zone_id', 'SlaveZoneId': 'ex_slave_zone_id'}
        for key in balancer_keys:
            params[key] = str(self.test.extra[balancer_keys[key]])
        self.assertUrlContainsQueryParams(url, params)
        body = self.fixtures.load('create_load_balancer.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _AddBackendServers(self, method, url, body, headers):
        _id = self.test.members[0].id
        self.assertTrue('ServerId' in url and _id in url)
        self.assertTrue('Weight' in url and '100' in url)
        body = self.fixtures.load('add_backend_servers.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _CreateLoadBalancerHTTPListener(self, method, url, body, headers):
        body = self.fixtures.load('create_load_balancer_http_listener.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _StartLoadBalancerListener(self, method, url, body, headers):
        body = self.fixtures.load('start_load_balancer_listener.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _RemoveBackendServers(self, method, url, body, headers):
        _id = self.test.member.id
        servers_json = '["%s"]' % _id
        params = {'LoadBalancerId': self.test.balancer.id, 'BackendServers': servers_json}
        self.assertUrlContainsQueryParams(url, params)
        body = self.fixtures.load('add_backend_servers.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _attach_compute_node_DescribeLoadBalancers(self, method, url, body, headers):
        body = self.fixtures.load('describe_load_balancers.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _attach_compute_node_AddBackendServers(self, method, url, body, headers):
        _id = self.test.node.id
        self.assertTrue('ServerId' in url and _id in url)
        self.assertTrue('Weight' in url and '100' in url)
        body = self.fixtures.load('add_backend_servers.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _create_listener_CreateLoadBalancerHTTPListener(self, method, url, body, headers):
        params = {'LoadBalancerId': self.test.balancer.id, 'RegionId': self.test.region, 'ListenerPort': str(self.test.balancer.port), 'BackendServerPort': str(self.test.backend_port), 'Scheduler': 'wrr', 'Bandwidth': '1', 'StickySession': 'off', 'HealthCheck': 'off'}
        self.assertUrlContainsQueryParams(url, params)
        body = self.fixtures.load('create_load_balancer_http_listener.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _create_listener_DescribeLoadBalancers(self, method, url, body, headers):
        body = self.fixtures.load('describe_load_balancers.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _create_listener_override_port_CreateLoadBalancerHTTPListener(self, method, url, body, headers):
        params = {'LoadBalancerId': self.test.balancer.id, 'RegionId': self.test.region, 'ListenerPort': str(self.test.extra['ListenerPort']), 'BackendServerPort': str(self.test.backend_port), 'Scheduler': 'wrr', 'Bandwidth': '1', 'StickySession': 'off', 'HealthCheck': 'off'}
        self.assertUrlContainsQueryParams(url, params)
        body = self.fixtures.load('create_load_balancer_http_listener.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _create_listener_override_port_DescribeLoadBalancers(self, method, url, body, headers):
        body = self.fixtures.load('describe_load_balancers.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _start_listener_DescribeLoadBalancers(self, method, url, body, headers):
        return self._DescribeLoadBalancers(method, url, body, headers)

    def _start_listener_StartLoadBalancerListener(self, method, url, body, headers):
        params = {'ListenerPort': str(self.test.port)}
        self.assertUrlContainsQueryParams(url, params)
        return self._StartLoadBalancerListener(method, url, body, headers)

    def _stop_listener_DescribeLoadBalancers(self, method, url, body, headers):
        return self._DescribeLoadBalancers(method, url, body, headers)

    def _stop_listener_StopLoadBalancerListener(self, method, url, body, headers):
        params = {'ListenerPort': str(self.test.port)}
        self.assertUrlContainsQueryParams(url, params)
        return self._StartLoadBalancerListener(method, url, body, headers)

    def _UploadServerCertificate(self, method, url, body, headers):
        body = self.fixtures.load('upload_server_certificate.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _DescribeServerCertificates(self, method, url, body, headers):
        body = self.fixtures.load('describe_server_certificates.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _list_certificates_ids_DescribeServerCertificates(self, method, url, body, headers):
        params = {'RegionId': self.test.region, 'ServerCertificateId': ','.join(self.test.cert_ids)}
        self.assertUrlContainsQueryParams(url, params)
        body = self.fixtures.load('describe_server_certificates.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _DeleteServerCertificate(self, method, url, body, headers):
        params = {'RegionId': self.test.region, 'ServerCertificateId': self.test.cert_id}
        self.assertUrlContainsQueryParams(url, params)
        body = self.fixtures.load('delete_server_certificate.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _SetServerCertificateName(self, method, url, body, headers):
        params = {'RegionId': self.test.region, 'ServerCertificateId': self.test.cert_id, 'ServerCertificateName': self.test.cert_name}
        self.assertUrlContainsQueryParams(url, params)
        body = self.fixtures.load('set_server_certificate_name.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])