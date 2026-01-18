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
class TestChangeResourceRecordSetsRoute53(AWSMockServiceTestCase):
    connection_class = Route53Connection

    def setUp(self):
        super(TestChangeResourceRecordSetsRoute53, self).setUp()

    def default_body(self):
        return b'\n<ChangeResourceRecordSetsResponse xmlns="https://route53.amazonaws.com/doc/2013-04-01/">\n    <ChangeInfo>\n        <Id>/change/C1111111111111</Id>\n        <Status>PENDING</Status>\n        <SubmittedAt>2014-05-05T10:11:12.123Z</SubmittedAt>\n    </ChangeInfo>\n</ChangeResourceRecordSetsResponse>\n        '

    def test_record_commit(self):
        rrsets = ResourceRecordSets(self.service_connection)
        rrsets.add_change_record('CREATE', Record('vanilla.example.com', 'A', 60, ['1.2.3.4']))
        rrsets.add_change_record('CREATE', Record('alias.example.com', 'AAAA', alias_hosted_zone_id='Z123OTHER', alias_dns_name='target.other', alias_evaluate_target_health=True))
        rrsets.add_change_record('CREATE', Record('wrr.example.com', 'CNAME', 60, ['cname.target'], weight=10, identifier='weight-1'))
        rrsets.add_change_record('CREATE', Record('lbr.example.com', 'TXT', 60, ['text record'], region='us-west-2', identifier='region-1'))
        rrsets.add_change_record('CREATE', Record('failover.example.com', 'A', 60, ['2.2.2.2'], health_check='hc-1234', failover='PRIMARY', identifier='primary'))
        changes_xml = rrsets.to_xml()
        actual_xml = re.sub('\\s*[\\r\\n]+', '\n', xml.dom.minidom.parseString(changes_xml).toprettyxml())
        expected_xml = re.sub('\\s*[\\r\\n]+', '\n', xml.dom.minidom.parseString(b'\n<ChangeResourceRecordSetsRequest xmlns="https://route53.amazonaws.com/doc/2013-04-01/">\n    <ChangeBatch>\n        <Comment>None</Comment>\n        <Changes>\n            <Change>\n                <Action>CREATE</Action>\n                <ResourceRecordSet>\n                    <Name>vanilla.example.com</Name>\n                    <Type>A</Type>\n                    <TTL>60</TTL>\n                    <ResourceRecords>\n                        <ResourceRecord>\n                            <Value>1.2.3.4</Value>\n                        </ResourceRecord>\n                    </ResourceRecords>\n                </ResourceRecordSet>\n            </Change>\n            <Change>\n                <Action>CREATE</Action>\n                <ResourceRecordSet>\n                    <Name>alias.example.com</Name>\n                    <Type>AAAA</Type>\n                    <AliasTarget>\n                        <HostedZoneId>Z123OTHER</HostedZoneId>\n                        <DNSName>target.other</DNSName>\n                        <EvaluateTargetHealth>true</EvaluateTargetHealth>\n                    </AliasTarget>\n                </ResourceRecordSet>\n            </Change>\n            <Change>\n                <Action>CREATE</Action>\n                <ResourceRecordSet>\n                    <Name>wrr.example.com</Name>\n                    <Type>CNAME</Type>\n                    <SetIdentifier>weight-1</SetIdentifier>\n                    <Weight>10</Weight>\n                    <TTL>60</TTL>\n                    <ResourceRecords>\n                        <ResourceRecord>\n                            <Value>cname.target</Value>\n                        </ResourceRecord>\n                    </ResourceRecords>\n                </ResourceRecordSet>\n            </Change>\n            <Change>\n                <Action>CREATE</Action>\n                <ResourceRecordSet>\n                    <Name>lbr.example.com</Name>\n                    <Type>TXT</Type>\n                    <SetIdentifier>region-1</SetIdentifier>\n                    <Region>us-west-2</Region>\n                    <TTL>60</TTL>\n                    <ResourceRecords>\n                        <ResourceRecord>\n                            <Value>text record</Value>\n                        </ResourceRecord>\n                    </ResourceRecords>\n                </ResourceRecordSet>\n            </Change>\n            <Change>\n                <Action>CREATE</Action>\n                <ResourceRecordSet>\n                    <Name>failover.example.com</Name>\n                    <Type>A</Type>\n                    <SetIdentifier>primary</SetIdentifier>\n                    <Failover>PRIMARY</Failover>\n                    <TTL>60</TTL>\n                    <ResourceRecords>\n                        <ResourceRecord>\n                            <Value>2.2.2.2</Value>\n                        </ResourceRecord>\n                    </ResourceRecords>\n                    <HealthCheckId>hc-1234</HealthCheckId>\n                </ResourceRecordSet>\n            </Change>\n        </Changes>\n    </ChangeBatch>\n</ChangeResourceRecordSetsRequest>\n        ').toprettyxml())
        self.assertEqual(actual_xml, expected_xml)