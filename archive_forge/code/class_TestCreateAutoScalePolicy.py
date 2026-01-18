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
class TestCreateAutoScalePolicy(AWSMockServiceTestCase):
    connection_class = AutoScaleConnection

    def setUp(self):
        super(TestCreateAutoScalePolicy, self).setUp()

    def default_body(self):
        return b'\n            <PutScalingPolicyResponse xmlns="http://autoscaling.amazonaws.com            /doc/2011-01-01/">\n              <PutScalingPolicyResult>\n                <PolicyARN>arn:aws:autoscaling:us-east-1:803981987763:scaling                Policy:b0dcf5e8\n            -02e6-4e31-9719-0675d0dc31ae:autoScalingGroupName/my-test-asg:            policyName/my-scal\n            eout-policy</PolicyARN>\n              </PutScalingPolicyResult>\n              <ResponseMetadata>\n                <RequestId>3cfc6fef-c08b-11e2-a697-2922EXAMPLE</RequestId>\n              </ResponseMetadata>\n            </PutScalingPolicyResponse>\n        '

    def test_scaling_policy_with_min_adjustment_step(self):
        self.set_http_response(status_code=200)
        policy = ScalingPolicy(name='foo', as_name='bar', adjustment_type='PercentChangeInCapacity', scaling_adjustment=50, min_adjustment_step=30)
        self.service_connection.create_scaling_policy(policy)
        self.assert_request_parameters({'Action': 'PutScalingPolicy', 'PolicyName': 'foo', 'AutoScalingGroupName': 'bar', 'AdjustmentType': 'PercentChangeInCapacity', 'ScalingAdjustment': 50, 'MinAdjustmentStep': 30}, ignore_params_values=['Version'])

    def test_scaling_policy_with_wrong_adjustment_type(self):
        self.set_http_response(status_code=200)
        policy = ScalingPolicy(name='foo', as_name='bar', adjustment_type='ChangeInCapacity', scaling_adjustment=50, min_adjustment_step=30)
        self.service_connection.create_scaling_policy(policy)
        self.assert_request_parameters({'Action': 'PutScalingPolicy', 'PolicyName': 'foo', 'AutoScalingGroupName': 'bar', 'AdjustmentType': 'ChangeInCapacity', 'ScalingAdjustment': 50}, ignore_params_values=['Version'])

    def test_scaling_policy_without_min_adjustment_step(self):
        self.set_http_response(status_code=200)
        policy = ScalingPolicy(name='foo', as_name='bar', adjustment_type='PercentChangeInCapacity', scaling_adjustment=50)
        self.service_connection.create_scaling_policy(policy)
        self.assert_request_parameters({'Action': 'PutScalingPolicy', 'PolicyName': 'foo', 'AutoScalingGroupName': 'bar', 'AdjustmentType': 'PercentChangeInCapacity', 'ScalingAdjustment': 50}, ignore_params_values=['Version'])