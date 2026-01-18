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
class TestGetAllRRSetsRoute53(AWSMockServiceTestCase):
    connection_class = Route53Connection

    def setUp(self):
        super(TestGetAllRRSetsRoute53, self).setUp()

    def default_body(self):
        return b'\n<ListResourceRecordSetsResponse xmlns="https://route53.amazonaws.com/doc/2013-04-01/">\n    <ResourceRecordSets>\n        <ResourceRecordSet>\n            <Name>test.example.com.</Name>\n            <Type>A</Type>\n            <TTL>60</TTL>\n            <ResourceRecords>\n                <ResourceRecord>\n                    <Value>10.0.0.1</Value>\n                </ResourceRecord>\n            </ResourceRecords>\n        </ResourceRecordSet>\n        <ResourceRecordSet>\n            <Name>www.example.com.</Name>\n            <Type>A</Type>\n            <TTL>60</TTL>\n            <ResourceRecords>\n                <ResourceRecord>\n                    <Value>10.0.0.2</Value>\n                </ResourceRecord>\n            </ResourceRecords>\n        </ResourceRecordSet>\n        <ResourceRecordSet>\n            <Name>us-west-2-evaluate-health.example.com.</Name>\n            <Type>A</Type>\n            <SetIdentifier>latency-example-us-west-2-evaluate-health</SetIdentifier>\n            <Region>us-west-2</Region>\n            <AliasTarget>\n                <HostedZoneId>ABCDEFG123456</HostedZoneId>\n                <EvaluateTargetHealth>true</EvaluateTargetHealth>\n                <DNSName>example-123456-evaluate-health.us-west-2.elb.amazonaws.com.</DNSName>\n            </AliasTarget>\n            <HealthCheckId>abcdefgh-abcd-abcd-abcd-abcdefghijkl</HealthCheckId>\n        </ResourceRecordSet>\n        <ResourceRecordSet>\n            <Name>us-west-2-no-evaluate-health.example.com.</Name>\n            <Type>A</Type>\n            <SetIdentifier>latency-example-us-west-2-no-evaluate-health</SetIdentifier>\n            <Region>us-west-2</Region>\n            <AliasTarget>\n                <HostedZoneId>ABCDEFG567890</HostedZoneId>\n                <EvaluateTargetHealth>false</EvaluateTargetHealth>\n                <DNSName>example-123456-no-evaluate-health.us-west-2.elb.amazonaws.com.</DNSName>\n            </AliasTarget>\n            <HealthCheckId>abcdefgh-abcd-abcd-abcd-abcdefghijkl</HealthCheckId>\n        </ResourceRecordSet>\n        <ResourceRecordSet>\n            <Name>failover.example.com.</Name>\n            <Type>A</Type>\n            <SetIdentifier>failover-primary</SetIdentifier>\n            <Failover>PRIMARY</Failover>\n            <TTL>60</TTL>\n            <ResourceRecords>\n                <ResourceRecord>\n                    <Value>10.0.0.4</Value>\n                </ResourceRecord>\n            </ResourceRecords>\n        </ResourceRecordSet>\n        <ResourceRecordSet>\n            <Name>us-west-2-evaluate-health-healthcheck.example.com.</Name>\n            <Type>A</Type>\n            <SetIdentifier>latency-example-us-west-2-evaluate-health-healthcheck</SetIdentifier>\n            <Region>us-west-2</Region>\n            <AliasTarget>\n                <HostedZoneId>ABCDEFG123456</HostedZoneId>\n                <EvaluateTargetHealth>true</EvaluateTargetHealth>\n                <DNSName>example-123456-evaluate-health-healthcheck.us-west-2.elb.amazonaws.com.</DNSName>\n            </AliasTarget>\n            <HealthCheckId>076a32f8-86f7-4c9e-9fa2-c163d5be67d9</HealthCheckId>\n        </ResourceRecordSet>\n    </ResourceRecordSets>\n    <IsTruncated>false</IsTruncated>\n    <MaxItems>100</MaxItems>\n</ListResourceRecordSetsResponse>\n        '

    def test_get_all_rr_sets(self):
        self.set_http_response(status_code=200)
        response = self.service_connection.get_all_rrsets('Z1111', 'A', 'example.com.')
        self.assertIn(self.actual_request.path, ('/2013-04-01/hostedzone/Z1111/rrset?type=A&name=example.com.', '/2013-04-01/hostedzone/Z1111/rrset?name=example.com.&type=A'))
        self.assertTrue(isinstance(response, ResourceRecordSets))
        self.assertEqual(response.hosted_zone_id, 'Z1111')
        self.assertTrue(isinstance(response[0], Record))
        self.assertTrue(response[0].name, 'test.example.com.')
        self.assertTrue(response[0].ttl, '60')
        self.assertTrue(response[0].type, 'A')
        evaluate_record = response[2]
        self.assertEqual(evaluate_record.name, 'us-west-2-evaluate-health.example.com.')
        self.assertEqual(evaluate_record.type, 'A')
        self.assertEqual(evaluate_record.identifier, 'latency-example-us-west-2-evaluate-health')
        self.assertEqual(evaluate_record.region, 'us-west-2')
        self.assertEqual(evaluate_record.alias_hosted_zone_id, 'ABCDEFG123456')
        self.assertTrue(evaluate_record.alias_evaluate_target_health)
        self.assertEqual(evaluate_record.alias_dns_name, 'example-123456-evaluate-health.us-west-2.elb.amazonaws.com.')
        evaluate_xml = evaluate_record.to_xml()
        self.assertTrue(evaluate_record.health_check, 'abcdefgh-abcd-abcd-abcd-abcdefghijkl')
        self.assertTrue('<EvaluateTargetHealth>true</EvaluateTargetHealth>' in evaluate_xml)
        no_evaluate_record = response[3]
        self.assertEqual(no_evaluate_record.name, 'us-west-2-no-evaluate-health.example.com.')
        self.assertEqual(no_evaluate_record.type, 'A')
        self.assertEqual(no_evaluate_record.identifier, 'latency-example-us-west-2-no-evaluate-health')
        self.assertEqual(no_evaluate_record.region, 'us-west-2')
        self.assertEqual(no_evaluate_record.alias_hosted_zone_id, 'ABCDEFG567890')
        self.assertFalse(no_evaluate_record.alias_evaluate_target_health)
        self.assertEqual(no_evaluate_record.alias_dns_name, 'example-123456-no-evaluate-health.us-west-2.elb.amazonaws.com.')
        no_evaluate_xml = no_evaluate_record.to_xml()
        self.assertTrue(no_evaluate_record.health_check, 'abcdefgh-abcd-abcd-abcd-abcdefghijkl')
        self.assertTrue('<EvaluateTargetHealth>false</EvaluateTargetHealth>' in no_evaluate_xml)
        failover_record = response[4]
        self.assertEqual(failover_record.name, 'failover.example.com.')
        self.assertEqual(failover_record.type, 'A')
        self.assertEqual(failover_record.identifier, 'failover-primary')
        self.assertEqual(failover_record.failover, 'PRIMARY')
        self.assertEqual(failover_record.ttl, '60')
        healthcheck_record = response[5]
        self.assertEqual(healthcheck_record.health_check, '076a32f8-86f7-4c9e-9fa2-c163d5be67d9')
        self.assertEqual(healthcheck_record.name, 'us-west-2-evaluate-health-healthcheck.example.com.')
        self.assertEqual(healthcheck_record.identifier, 'latency-example-us-west-2-evaluate-health-healthcheck')
        self.assertEqual(healthcheck_record.alias_dns_name, 'example-123456-evaluate-health-healthcheck.us-west-2.elb.amazonaws.com.')