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
class TestLaunchConfigurationDescribeWithBlockDeviceTypes(AWSMockServiceTestCase):
    connection_class = AutoScaleConnection

    def default_body(self):
        return b'\n        <DescribeLaunchConfigurationsResponse>\n          <DescribeLaunchConfigurationsResult>\n            <LaunchConfigurations>\n              <member>\n                <AssociatePublicIpAddress>true</AssociatePublicIpAddress>\n                <SecurityGroups/>\n                <CreatedTime>2013-01-21T23:04:42.200Z</CreatedTime>\n                <KernelId/>\n                <LaunchConfigurationName>my-test-lc</LaunchConfigurationName>\n                <UserData/>\n                <InstanceType>m1.small</InstanceType>\n                <LaunchConfigurationARN>arn:aws:autoscaling:us-east-1:803981987763:launchConfiguration:9dbbbf87-6141-428a-a409-0752edbe6cad:launchConfigurationName/my-test-lc</LaunchConfigurationARN>\n                <BlockDeviceMappings>\n                  <member>\n                    <DeviceName>/dev/xvdp</DeviceName>\n                    <Ebs>\n                      <SnapshotId>snap-1234abcd</SnapshotId>\n                      <Iops>1000</Iops>\n                      <DeleteOnTermination>true</DeleteOnTermination>\n                      <VolumeType>io1</VolumeType>\n                      <VolumeSize>100</VolumeSize>\n                    </Ebs>\n                  </member>\n                  <member>\n                    <VirtualName>ephemeral1</VirtualName>\n                    <DeviceName>/dev/xvdc</DeviceName>\n                  </member>\n                  <member>\n                    <VirtualName>ephemeral0</VirtualName>\n                    <DeviceName>/dev/xvdb</DeviceName>\n                  </member>\n                  <member>\n                    <DeviceName>/dev/xvdh</DeviceName>\n                    <Ebs>\n                      <Iops>2000</Iops>\n                      <DeleteOnTermination>false</DeleteOnTermination>\n                      <VolumeType>io1</VolumeType>\n                      <VolumeSize>200</VolumeSize>\n                    </Ebs>\n                  </member>\n                </BlockDeviceMappings>\n                <ImageId>ami-514ac838</ImageId>\n                <KeyName/>\n                <RamdiskId/>\n                <InstanceMonitoring>\n                  <Enabled>true</Enabled>\n                </InstanceMonitoring>\n                <EbsOptimized>false</EbsOptimized>\n              </member>\n            </LaunchConfigurations>\n          </DescribeLaunchConfigurationsResult>\n          <ResponseMetadata>\n            <RequestId>d05a22f8-b690-11e2-bf8e-2113fEXAMPLE</RequestId>\n          </ResponseMetadata>\n        </DescribeLaunchConfigurationsResponse>\n        '

    def test_get_all_launch_configurations_with_block_device_types(self):
        self.set_http_response(status_code=200)
        self.service_connection.use_block_device_types = True
        response = self.service_connection.get_all_launch_configurations()
        self.assertTrue(isinstance(response, list))
        self.assertEqual(len(response), 1)
        self.assertTrue(isinstance(response[0], LaunchConfiguration))
        self.assertEqual(response[0].associate_public_ip_address, True)
        self.assertEqual(response[0].name, 'my-test-lc')
        self.assertEqual(response[0].instance_type, 'm1.small')
        self.assertEqual(response[0].launch_configuration_arn, 'arn:aws:autoscaling:us-east-1:803981987763:launchConfiguration:9dbbbf87-6141-428a-a409-0752edbe6cad:launchConfigurationName/my-test-lc')
        self.assertEqual(response[0].image_id, 'ami-514ac838')
        self.assertTrue(isinstance(response[0].instance_monitoring, launchconfig.InstanceMonitoring))
        self.assertEqual(response[0].instance_monitoring.enabled, 'true')
        self.assertEqual(response[0].ebs_optimized, False)
        self.assertEqual(response[0].block_device_mappings['/dev/xvdb'].ephemeral_name, 'ephemeral0')
        self.assertEqual(response[0].block_device_mappings['/dev/xvdc'].ephemeral_name, 'ephemeral1')
        self.assertEqual(response[0].block_device_mappings['/dev/xvdp'].snapshot_id, 'snap-1234abcd')
        self.assertEqual(response[0].block_device_mappings['/dev/xvdp'].delete_on_termination, True)
        self.assertEqual(response[0].block_device_mappings['/dev/xvdp'].iops, 1000)
        self.assertEqual(response[0].block_device_mappings['/dev/xvdp'].size, 100)
        self.assertEqual(response[0].block_device_mappings['/dev/xvdp'].volume_type, 'io1')
        self.assertEqual(response[0].block_device_mappings['/dev/xvdh'].delete_on_termination, False)
        self.assertEqual(response[0].block_device_mappings['/dev/xvdh'].iops, 2000)
        self.assertEqual(response[0].block_device_mappings['/dev/xvdh'].size, 200)
        self.assertEqual(response[0].block_device_mappings['/dev/xvdh'].volume_type, 'io1')
        self.assert_request_parameters({'Action': 'DescribeLaunchConfigurations'}, ignore_params_values=['Version'])

    def test_get_all_configuration_limited(self):
        self.set_http_response(status_code=200)
        response = self.service_connection.get_all_launch_configurations(max_records=10, names=['my-test1', 'my-test2'])
        self.assert_request_parameters({'Action': 'DescribeLaunchConfigurations', 'MaxRecords': 10, 'LaunchConfigurationNames.member.1': 'my-test1', 'LaunchConfigurationNames.member.2': 'my-test2'}, ignore_params_values=['Version'])