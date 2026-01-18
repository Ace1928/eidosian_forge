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
class TestDeleteNotificationConfiguration(AWSMockServiceTestCase):
    connection_class = AutoScaleConnection

    def setUp(self):
        super(TestDeleteNotificationConfiguration, self).setUp()

    def default_body(self):
        return b'\n            <DeleteNotificationConfigurationResponse>\n              <ResponseMetadata>\n                <RequestId>requestid</RequestId>\n              </ResponseMetadata>\n            </DeleteNotificationConfigurationResponse>\n        '

    def test_autoscaling_group_put_notification_configuration(self):
        self.set_http_response(status_code=200)
        autoscale = AutoScalingGroup(name='ana', launch_config='lauch_config', min_size=1, max_size=2, termination_policies=['OldestInstance', 'OldestLaunchConfiguration'])
        self.service_connection.delete_notification_configuration(autoscale, 'arn:aws:sns:us-east-1:19890506:AutoScaling-Up')
        self.assert_request_parameters({'Action': 'DeleteNotificationConfiguration', 'AutoScalingGroupName': 'ana', 'TopicARN': 'arn:aws:sns:us-east-1:19890506:AutoScaling-Up'}, ignore_params_values=['Version'])