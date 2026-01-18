import copy
import json
from heatclient import exc
from oslo_log import log as logging
from testtools import matchers
from heat_integrationtests.common import test
from heat_integrationtests.functional import functional_base
class AutoscalingGroupTest(functional_base.FunctionalTestsBase):
    template = '\n{\n  "AWSTemplateFormatVersion" : "2010-09-09",\n  "Description" : "Template to create multiple instances.",\n  "Parameters" : {"size": {"Type": "String", "Default": "1"},\n                  "AZ": {"Type": "String", "Default": "nova"},\n                  "image": {"Type": "String"},\n                  "flavor": {"Type": "String"},\n                  "user_data": {"Type": "String", "Default": "jsconfig data"}},\n  "Resources": {\n    "JobServerGroup": {\n      "Type" : "AWS::AutoScaling::AutoScalingGroup",\n      "Properties" : {\n        "AvailabilityZones" : [{"Ref": "AZ"}],\n        "LaunchConfigurationName" : { "Ref" : "JobServerConfig" },\n        "MinSize" : {"Ref": "size"},\n        "MaxSize" : "20"\n      }\n    },\n\n    "JobServerConfig" : {\n      "Type" : "AWS::AutoScaling::LaunchConfiguration",\n      "Metadata": {"foo": "bar"},\n      "Properties": {\n        "ImageId"           : {"Ref": "image"},\n        "InstanceType"      : {"Ref": "flavor"},\n        "SecurityGroups"    : [ "sg-1" ],\n        "UserData"          : {"Ref": "user_data"}\n      }\n    }\n  },\n  "Outputs": {\n    "InstanceList": {"Value": {\n      "Fn::GetAtt": ["JobServerGroup", "InstanceList"]}},\n    "JobServerConfigRef": {"Value": {\n      "Ref": "JobServerConfig"}}\n  }\n}\n'
    instance_template = '\nheat_template_version: 2013-05-23\nparameters:\n  ImageId: {type: string}\n  InstanceType: {type: string}\n  SecurityGroups: {type: comma_delimited_list}\n  UserData: {type: string}\n  Tags: {type: comma_delimited_list, default: "x,y"}\n\nresources:\n  random1:\n    type: OS::Heat::RandomString\n    properties:\n      salt: {get_param: UserData}\noutputs:\n  PublicIp: {value: {get_attr: [random1, value]}}\n  AvailabilityZone: {value: \'not-used11\'}\n  PrivateDnsName: {value: \'not-used12\'}\n  PublicDnsName: {value: \'not-used13\'}\n  PrivateIp: {value: \'not-used14\'}\n'
    bad_instance_template = '\nheat_template_version: 2013-05-23\nparameters:\n  ImageId: {type: string}\n  InstanceType: {type: string}\n  SecurityGroups: {type: comma_delimited_list}\n  UserData: {type: string}\n  Tags: {type: comma_delimited_list, default: "x,y"}\n\nresources:\n  random1:\n    type: OS::Heat::RandomString\n    depends_on: waiter\n  ready_poster:\n    type: AWS::CloudFormation::WaitConditionHandle\n  waiter:\n    type: AWS::CloudFormation::WaitCondition\n    properties:\n      Handle: {get_resource: ready_poster}\n      Timeout: 1\noutputs:\n  PublicIp:\n    value: {get_attr: [random1, value]}\n'

    def setUp(self):
        super(AutoscalingGroupTest, self).setUp()
        if not self.conf.minimal_image_ref:
            raise self.skipException('No minimal image configured to test')
        if not self.conf.instance_type:
            raise self.skipException('No flavor configured to test')

    def assert_instance_count(self, stack, expected_count):
        inst_list = self._stack_output(stack, 'InstanceList')
        self.assertEqual(expected_count, len(inst_list.split(',')))

    def _assert_instance_state(self, nested_identifier, num_complete, num_failed):
        for res in self.client.resources.list(nested_identifier):
            if 'COMPLETE' in res.resource_status:
                num_complete = num_complete - 1
            elif 'FAILED' in res.resource_status:
                num_failed = num_failed - 1
        self.assertEqual(0, num_failed)
        self.assertEqual(0, num_complete)