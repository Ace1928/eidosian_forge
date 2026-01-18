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
class TestModifyReservedInstances(TestEC2ConnectionBase):

    def default_body(self):
        return b"<ModifyReservedInstancesResponse xmlns='http://ec2.amazonaws.com/doc/2013-08-15/'>\n    <requestId>bef729b6-0731-4489-8881-2258746ae163</requestId>\n    <reservedInstancesModificationId>rimod-3aae219d-3d63-47a9-a7e9-e764example</reservedInstancesModificationId>\n</ModifyReservedInstancesResponse>"

    def test_serialized_api_args(self):
        self.set_http_response(status_code=200)
        response = self.ec2.modify_reserved_instances('a-token-goes-here', reserved_instance_ids=['2567o137-8a55-48d6-82fb-7258506bb497'], target_configurations=[ReservedInstancesConfiguration(availability_zone='us-west-2c', platform='EC2-VPC', instance_count=3, instance_type='c3.large')])
        self.assert_request_parameters({'Action': 'ModifyReservedInstances', 'ClientToken': 'a-token-goes-here', 'ReservedInstancesConfigurationSetItemType.0.AvailabilityZone': 'us-west-2c', 'ReservedInstancesConfigurationSetItemType.0.InstanceCount': 3, 'ReservedInstancesConfigurationSetItemType.0.Platform': 'EC2-VPC', 'ReservedInstancesConfigurationSetItemType.0.InstanceType': 'c3.large', 'ReservedInstancesId.1': '2567o137-8a55-48d6-82fb-7258506bb497'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
        self.assertEqual(response, 'rimod-3aae219d-3d63-47a9-a7e9-e764example')

    def test_none_token(self):
        """Ensures that if the token is set to None, nothing is serialized."""
        self.set_http_response(status_code=200)
        response = self.ec2.modify_reserved_instances(None, reserved_instance_ids=['2567o137-8a55-48d6-82fb-7258506bb497'], target_configurations=[ReservedInstancesConfiguration(availability_zone='us-west-2c', platform='EC2-VPC', instance_count=3, instance_type='c3.large')])
        self.assert_request_parameters({'Action': 'ModifyReservedInstances', 'ReservedInstancesConfigurationSetItemType.0.AvailabilityZone': 'us-west-2c', 'ReservedInstancesConfigurationSetItemType.0.InstanceCount': 3, 'ReservedInstancesConfigurationSetItemType.0.Platform': 'EC2-VPC', 'ReservedInstancesConfigurationSetItemType.0.InstanceType': 'c3.large', 'ReservedInstancesId.1': '2567o137-8a55-48d6-82fb-7258506bb497'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
        self.assertEqual(response, 'rimod-3aae219d-3d63-47a9-a7e9-e764example')