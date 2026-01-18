from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.ec2.securitygroup import SecurityGroup
from boto.rds import RDSConnection
from boto.rds.vpcsecuritygroupmembership import VPCSecurityGroupMembership
from boto.rds.parametergroup import ParameterGroup
from boto.rds.logfile import LogFile, LogFileObject
import xml.sax.saxutils as saxutils
class TestRDSOptionGroupOptions(AWSMockServiceTestCase):
    connection_class = RDSConnection

    def setUp(self):
        super(TestRDSOptionGroupOptions, self).setUp()

    def default_body(self):
        return '\n        <DescribeOptionGroupOptionsResponse xmlns="http://rds.amazonaws.com/doc/2013-05-15/">\n          <DescribeOptionGroupOptionsResult>\n            <OptionGroupOptions>\n              <OptionGroupOption>\n                <MajorEngineVersion>11.2</MajorEngineVersion>\n                <PortRequired>true</PortRequired>\n                <OptionsDependedOn/>\n                <Description>Oracle Enterprise Manager</Description>\n                <DefaultPort>1158</DefaultPort>\n                <Name>OEM</Name>\n                <EngineName>oracle-se1</EngineName>\n                <MinimumRequiredMinorEngineVersion>0.2.v3</MinimumRequiredMinorEngineVersion>\n                <Persistent>false</Persistent>\n                <Permanent>false</Permanent>\n              </OptionGroupOption>\n            </OptionGroupOptions>\n          </DescribeOptionGroupOptionsResult>\n          <ResponseMetadata>\n            <RequestId>d9c8f6a1-84c7-11e1-a264-0b23c28bc344</RequestId>\n          </ResponseMetadata>\n        </DescribeOptionGroupOptionsResponse>\n        '

    def test_describe_option_group_options(self):
        self.set_http_response(status_code=200)
        response = self.service_connection.describe_option_group_options()
        self.assertEqual(len(response), 1)
        options = response[0]
        self.assertEqual(options.name, 'OEM')
        self.assertEqual(options.description, 'Oracle Enterprise Manager')
        self.assertEqual(options.engine_name, 'oracle-se1')
        self.assertEqual(options.major_engine_version, '11.2')
        self.assertEqual(options.min_minor_engine_version, '0.2.v3')
        self.assertEqual(options.port_required, True)
        self.assertEqual(options.default_port, 1158)
        self.assertEqual(options.permanent, False)
        self.assertEqual(options.persistent, False)
        self.assertEqual(options.depends_on, [])