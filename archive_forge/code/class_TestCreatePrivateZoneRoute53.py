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
class TestCreatePrivateZoneRoute53(AWSMockServiceTestCase):
    connection_class = Route53Connection

    def setUp(self):
        super(TestCreatePrivateZoneRoute53, self).setUp()

    def default_body(self):
        return b'\n<CreateHostedZoneResponse xmlns="https://route53.amazonaws.com/doc/2012-02-29/">\n    <HostedZone>\n        <Id>/hostedzone/Z11111</Id>\n        <Name>example.com.</Name>\n        <VPC>\n           <VPCId>vpc-1a2b3c4d</VPCId>\n           <VPCRegion>us-east-1</VPCRegion>\n        </VPC>\n        <CallerReference>aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee</CallerReference>\n        <Config>\n            <Comment></Comment>\n            <PrivateZone>true</PrivateZone>\n        </Config>\n        <ResourceRecordSetCount>2</ResourceRecordSetCount>\n    </HostedZone>\n    <ChangeInfo>\n        <Id>/change/C1111111111111</Id>\n        <Status>PENDING</Status>\n        <SubmittedAt>2014-02-02T10:19:29.928Z</SubmittedAt>\n    </ChangeInfo>\n    <DelegationSet>\n        <NameServers>\n            <NameServer>ns-100.awsdns-01.com</NameServer>\n            <NameServer>ns-1000.awsdns-01.co.uk</NameServer>\n            <NameServer>ns-1000.awsdns-01.org</NameServer>\n            <NameServer>ns-900.awsdns-01.net</NameServer>\n        </NameServers>\n    </DelegationSet>\n</CreateHostedZoneResponse>\n        '

    def test_create_private_zone(self):
        self.set_http_response(status_code=201)
        r = self.service_connection.create_hosted_zone('example.com.', private_zone=True, vpc_id='vpc-1a2b3c4d', vpc_region='us-east-1')
        self.assertEqual(r['CreateHostedZoneResponse']['HostedZone']['Config']['PrivateZone'], u'true')
        self.assertEqual(r['CreateHostedZoneResponse']['HostedZone']['VPC']['VPCId'], u'vpc-1a2b3c4d')
        self.assertEqual(r['CreateHostedZoneResponse']['HostedZone']['VPC']['VPCRegion'], u'us-east-1')