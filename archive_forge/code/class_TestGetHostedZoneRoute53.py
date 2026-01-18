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
class TestGetHostedZoneRoute53(AWSMockServiceTestCase):
    connection_class = Route53Connection

    def setUp(self):
        super(TestGetHostedZoneRoute53, self).setUp()

    def default_body(self):
        return b'\n<GetHostedZoneResponse xmlns="https://route53.amazonaws.com/doc/2012-02-29/">\n    <HostedZone>\n        <Id>/hostedzone/Z1111</Id>\n        <Name>example.com.</Name>\n        <CallerReference>aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee</CallerReference>\n        <Config/>\n        <ResourceRecordSetCount>3</ResourceRecordSetCount>\n    </HostedZone>\n    <DelegationSet>\n        <NameServers>\n            <NameServer>ns-1000.awsdns-40.org</NameServer>\n            <NameServer>ns-200.awsdns-30.com</NameServer>\n            <NameServer>ns-900.awsdns-50.net</NameServer>\n            <NameServer>ns-1000.awsdns-00.co.uk</NameServer>\n        </NameServers>\n    </DelegationSet>\n</GetHostedZoneResponse>\n'

    def test_list_zones(self):
        self.set_http_response(status_code=201)
        response = self.service_connection.get_hosted_zone('Z1111')
        self.assertEqual(response['GetHostedZoneResponse']['HostedZone']['Id'], '/hostedzone/Z1111')
        self.assertEqual(response['GetHostedZoneResponse']['HostedZone']['Name'], 'example.com.')
        self.assertEqual(response['GetHostedZoneResponse']['DelegationSet']['NameServers'], ['ns-1000.awsdns-40.org', 'ns-200.awsdns-30.com', 'ns-900.awsdns-50.net', 'ns-1000.awsdns-00.co.uk'])