from datetime import datetime, timedelta
from mock import MagicMock, Mock
from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
import boto.ec2
from boto.regioninfo import RegionInfo
from boto.ec2.blockdevicemapping import BlockDeviceType, BlockDeviceMapping
from boto.ec2.connection import EC2Connection
from boto.ec2.snapshot import Snapshot
from boto.ec2.reservedinstance import ReservedInstancesConfiguration
from boto.compat import http_client
class TestGetClassicLinkInstances(TestEC2ConnectionBase):

    def default_body(self):
        return b'\n            <DescribeClassicLinkInstancesResponse xmlns="http://ec2.amazonaws.com/doc/2014-09-01/">\n               <requestId>f4bf0cc6-5967-4687-9355-90ce48394bd3</requestId>\n               <instancesSet>\n                  <item>\n                     <instanceId>i-31489bd8</instanceId>\n                     <vpcId>vpc-9d24f8f8</vpcId>\n                     <groupSet>\n                        <item>\n                           <groupId>sg-9b4343fe</groupId>\n                        </item>\n                    </groupSet>\n                    <tagSet>\n                        <item>\n                           <key>Name</key>\n                           <value>hello</value>\n                        </item>\n                    </tagSet>\n                 </item>\n              </instancesSet>\n           </DescribeClassicLinkInstancesResponse>\n        '

    def test_get_classic_link_instances(self):
        self.set_http_response(status_code=200)
        response = self.ec2.get_all_classic_link_instances()
        self.assertEqual(len(response), 1)
        instance = response[0]
        self.assertEqual(instance.id, 'i-31489bd8')
        self.assertEqual(instance.vpc_id, 'vpc-9d24f8f8')
        self.assertEqual(len(instance.groups), 1)
        self.assertEqual(instance.groups[0].id, 'sg-9b4343fe')
        self.assertEqual(instance.tags, {'Name': 'hello'})

    def test_get_classic_link_instances_params(self):
        self.set_http_response(status_code=200)
        self.ec2.get_all_classic_link_instances(instance_ids=['id1', 'id2'], filters={'GroupId': 'sg-9b4343fe'}, dry_run=True, next_token='next_token', max_results=10)
        self.assert_request_parameters({'Action': 'DescribeClassicLinkInstances', 'InstanceId.1': 'id1', 'InstanceId.2': 'id2', 'Filter.1.Name': 'GroupId', 'Filter.1.Value.1': 'sg-9b4343fe', 'DryRun': 'true', 'NextToken': 'next_token', 'MaxResults': 10}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])