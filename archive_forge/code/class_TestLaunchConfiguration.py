import base64
from datetime import datetime
from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.ec2.autoscale import AutoScaleConnection
from boto.ec2.autoscale.group import AutoScalingGroup
from boto.ec2.autoscale.policy import ScalingPolicy
from boto.ec2.autoscale.tag import Tag
from boto.ec2.blockdevicemapping import EBSBlockDeviceType, BlockDeviceMapping
from boto.ec2.autoscale import launchconfig, LaunchConfiguration
class TestLaunchConfiguration(AWSMockServiceTestCase):
    connection_class = AutoScaleConnection

    def default_body(self):
        return b'\n        <DescribeLaunchConfigurationsResponse>\n        </DescribeLaunchConfigurationsResponse>\n        '

    def test_launch_config(self):
        self.set_http_response(status_code=200)
        dev_sdf = EBSBlockDeviceType(snapshot_id='snap-12345')
        bdm = BlockDeviceMapping()
        bdm['/dev/sdf'] = dev_sdf
        lc = launchconfig.LaunchConfiguration(connection=self.service_connection, name='launch_config', image_id='123456', instance_type='m1.large', user_data='#!/bin/bash', security_groups=['group1'], spot_price='price', block_device_mappings=[bdm], associate_public_ip_address=True, volume_type='atype', delete_on_termination=False, iops=3000, classic_link_vpc_id='vpc-1234', classic_link_vpc_security_groups=['classic_link_group'])
        response = self.service_connection.create_launch_configuration(lc)
        self.assert_request_parameters({'Action': 'CreateLaunchConfiguration', 'BlockDeviceMappings.member.1.DeviceName': '/dev/sdf', 'BlockDeviceMappings.member.1.Ebs.DeleteOnTermination': 'false', 'BlockDeviceMappings.member.1.Ebs.SnapshotId': 'snap-12345', 'EbsOptimized': 'false', 'LaunchConfigurationName': 'launch_config', 'ImageId': '123456', 'UserData': base64.b64encode(b'#!/bin/bash').decode('utf-8'), 'InstanceMonitoring.Enabled': 'false', 'InstanceType': 'm1.large', 'SecurityGroups.member.1': 'group1', 'SpotPrice': 'price', 'AssociatePublicIpAddress': 'true', 'VolumeType': 'atype', 'DeleteOnTermination': 'false', 'Iops': 3000, 'ClassicLinkVPCId': 'vpc-1234', 'ClassicLinkVPCSecurityGroups.member.1': 'classic_link_group'}, ignore_params_values=['Version'])