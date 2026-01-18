from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.ec2.securitygroup import SecurityGroup
from boto.rds import RDSConnection
from boto.rds.vpcsecuritygroupmembership import VPCSecurityGroupMembership
from boto.rds.parametergroup import ParameterGroup
from boto.rds.logfile import LogFile, LogFileObject
import xml.sax.saxutils as saxutils
class TestRDSLogFile(AWSMockServiceTestCase):
    connection_class = RDSConnection

    def setUp(self):
        super(TestRDSLogFile, self).setUp()

    def default_body(self):
        return '\n        <DescribeDBLogFilesResponse xmlns="http://rds.amazonaws.com/doc/2013-02-12/">\n          <DescribeDBLogFilesResult>\n            <DescribeDBLogFiles>\n              <DescribeDBLogFilesDetails>\n                <LastWritten>1364403600000</LastWritten>\n                <LogFileName>error/mysql-error-running.log</LogFileName>\n                <Size>0</Size>\n              </DescribeDBLogFilesDetails>\n              <DescribeDBLogFilesDetails>\n                <LastWritten>1364338800000</LastWritten>\n                <LogFileName>error/mysql-error-running.log.0</LogFileName>\n                <Size>0</Size>\n              </DescribeDBLogFilesDetails>\n              <DescribeDBLogFilesDetails>\n                <LastWritten>1364342400000</LastWritten>\n                <LogFileName>error/mysql-error-running.log.1</LogFileName>\n                <Size>0</Size>\n              </DescribeDBLogFilesDetails>\n              <DescribeDBLogFilesDetails>\n                <LastWritten>1364346000000</LastWritten>\n                <LogFileName>error/mysql-error-running.log.2</LogFileName>\n                <Size>0</Size>\n              </DescribeDBLogFilesDetails>\n              <DescribeDBLogFilesDetails>\n                <LastWritten>1364349600000</LastWritten>\n                <LogFileName>error/mysql-error-running.log.3</LogFileName>\n                <Size>0</Size>\n              </DescribeDBLogFilesDetails>\n              <DescribeDBLogFilesDetails>\n                <LastWritten>1364405700000</LastWritten>\n                <LogFileName>error/mysql-error.log</LogFileName>\n                <Size>0</Size>\n              </DescribeDBLogFilesDetails>\n            </DescribeDBLogFiles>\n          </DescribeDBLogFilesResult>\n          <ResponseMetadata>\n            <RequestId>d70fb3b3-9704-11e2-a0db-871552e0ef19</RequestId>\n          </ResponseMetadata>\n        </DescribeDBLogFilesResponse>\n        '

    def test_get_all_logs_simple(self):
        self.set_http_response(status_code=200)
        response = self.service_connection.get_all_logs('db1')
        self.assert_request_parameters({'Action': 'DescribeDBLogFiles', 'DBInstanceIdentifier': 'db1'}, ignore_params_values=['Version'])
        self.assertEqual(len(response), 6)
        self.assertTrue(isinstance(response[0], LogFile))
        self.assertEqual(response[0].log_filename, 'error/mysql-error-running.log')
        self.assertEqual(response[0].last_written, '1364403600000')
        self.assertEqual(response[0].size, '0')

    def test_get_all_logs_filtered(self):
        self.set_http_response(status_code=200)
        response = self.service_connection.get_all_logs('db_instance_1', max_records=100, marker='error/mysql-error.log', file_size=2000000, filename_contains='error', file_last_written=12345678)
        self.assert_request_parameters({'Action': 'DescribeDBLogFiles', 'DBInstanceIdentifier': 'db_instance_1', 'MaxRecords': 100, 'Marker': 'error/mysql-error.log', 'FileSize': 2000000, 'FilenameContains': 'error', 'FileLastWritten': 12345678}, ignore_params_values=['Version'])
        self.assertEqual(len(response), 6)
        self.assertTrue(isinstance(response[0], LogFile))
        self.assertEqual(response[0].log_filename, 'error/mysql-error-running.log')
        self.assertEqual(response[0].last_written, '1364403600000')
        self.assertEqual(response[0].size, '0')