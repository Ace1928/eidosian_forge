from base64 import b64decode
from boto.compat import json
from boto.iam.connection import IAMConnection
from tests.unit import AWSMockServiceTestCase
class TestCreateRole(AWSMockServiceTestCase):
    connection_class = IAMConnection

    def default_body(self):
        return b'\n          <CreateRoleResponse xmlns="https://iam.amazonaws.com/doc/2010-05-08/">\n            <CreateRoleResult>\n              <Role>\n                <Path>/application_abc/component_xyz/</Path>\n                <Arn>arn:aws:iam::123456789012:role/application_abc/component_xyz/S3Access</Arn>\n                <RoleName>S3Access</RoleName>\n                <AssumeRolePolicyDocument>{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":["ec2.amazonaws.com"]},"Action":["sts:AssumeRole"]}]}</AssumeRolePolicyDocument>\n                <CreateDate>2012-05-08T23:34:01.495Z</CreateDate>\n                <RoleId>AROADBQP57FF2AEXAMPLE</RoleId>\n              </Role>\n            </CreateRoleResult>\n            <ResponseMetadata>\n              <RequestId>4a93ceee-9966-11e1-b624-b1aEXAMPLE7c</RequestId>\n            </ResponseMetadata>\n          </CreateRoleResponse>\n        '

    def test_create_role_default(self):
        self.set_http_response(status_code=200)
        self.service_connection.create_role('a_name')
        self.assert_request_parameters({'Action': 'CreateRole', 'RoleName': 'a_name'}, ignore_params_values=['Version', 'AssumeRolePolicyDocument'])
        self.assertDictEqual(json.loads(self.actual_request.params['AssumeRolePolicyDocument']), {'Statement': [{'Action': ['sts:AssumeRole'], 'Effect': 'Allow', 'Principal': {'Service': ['ec2.amazonaws.com']}}]})

    def test_create_role_default_cn_north(self):
        self.set_http_response(status_code=200)
        self.service_connection.host = 'iam.cn-north-1.amazonaws.com.cn'
        self.service_connection.create_role('a_name')
        self.assert_request_parameters({'Action': 'CreateRole', 'RoleName': 'a_name'}, ignore_params_values=['Version', 'AssumeRolePolicyDocument'])
        self.assertDictEqual(json.loads(self.actual_request.params['AssumeRolePolicyDocument']), {'Statement': [{'Action': ['sts:AssumeRole'], 'Effect': 'Allow', 'Principal': {'Service': ['ec2.amazonaws.com.cn']}}]})

    def test_create_role_string_policy(self):
        self.set_http_response(status_code=200)
        self.service_connection.create_role('a_name', assume_role_policy_document='{"hello": "policy"}')
        self.assert_request_parameters({'Action': 'CreateRole', 'AssumeRolePolicyDocument': '{"hello": "policy"}', 'RoleName': 'a_name'}, ignore_params_values=['Version'])

    def test_create_role_data_policy(self):
        self.set_http_response(status_code=200)
        self.service_connection.create_role('a_name', assume_role_policy_document={'hello': 'policy'})
        self.assert_request_parameters({'Action': 'CreateRole', 'AssumeRolePolicyDocument': '{"hello": "policy"}', 'RoleName': 'a_name'}, ignore_params_values=['Version'])