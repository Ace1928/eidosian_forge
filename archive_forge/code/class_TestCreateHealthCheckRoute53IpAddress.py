from tests.compat import mock
import re
import xml.dom.minidom
from boto.exception import BotoServerError
from boto.route53.connection import Route53Connection
from boto.route53.exception import DNSServerError
from boto.route53.healthcheck import HealthCheck
from boto.route53.record import ResourceRecordSets, Record
from boto.route53.zone import Zone
from nose.plugins.attrib import attr
from tests.unit import AWSMockServiceTestCase
from boto.compat import six
@attr(route53=True)
class TestCreateHealthCheckRoute53IpAddress(AWSMockServiceTestCase):
    connection_class = Route53Connection

    def setUp(self):
        super(TestCreateHealthCheckRoute53IpAddress, self).setUp()

    def default_body(self):
        return b'\n<CreateHealthCheckResponse xmlns="https://route53.amazonaws.com/doc/2013-04-01/">\n   <HealthCheck>\n      <Id>34778cf8-e31e-4974-bad0-b108bd1623d3</Id>\n      <CallerReference>2fa48c8f-76ef-4253-9874-8bcb2b0d7694</CallerReference>\n      <HealthCheckConfig>\n         <IPAddress>74.125.228.81</IPAddress>\n         <Port>443</Port>\n         <Type>HTTPS_STR_MATCH</Type>\n         <SearchString>OK</SearchString>\n         <ResourcePath>/health_check</ResourcePath>\n         <RequestInterval>30</RequestInterval>\n         <FailureThreshold>3</FailureThreshold>\n      </HealthCheckConfig>\n   </HealthCheck>\n</CreateHealthCheckResponse>\n        '

    def test_create_health_check_ip_address(self):
        self.set_http_response(status_code=201)
        hc = HealthCheck(ip_addr='74.125.228.81', port=443, hc_type='HTTPS_STR_MATCH', resource_path='/health_check', string_match='OK')
        hc_xml = hc.to_xml()
        self.assertFalse('<FullyQualifiedDomainName>' in hc_xml)
        self.assertTrue('<IPAddress>' in hc_xml)
        response = self.service_connection.create_health_check(hc)
        hc_resp = response['CreateHealthCheckResponse']['HealthCheck']['HealthCheckConfig']
        self.assertEqual(hc_resp['IPAddress'], '74.125.228.81')
        self.assertEqual(hc_resp['ResourcePath'], '/health_check')
        self.assertEqual(hc_resp['Type'], 'HTTPS_STR_MATCH')
        self.assertEqual(hc_resp['Port'], '443')
        self.assertEqual(hc_resp['ResourcePath'], '/health_check')
        self.assertEqual(hc_resp['SearchString'], 'OK')
        self.assertEqual(response['CreateHealthCheckResponse']['HealthCheck']['Id'], '34778cf8-e31e-4974-bad0-b108bd1623d3')