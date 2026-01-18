import unittest
from datetime import datetime
from mock import Mock
from tests.unit import AWSMockServiceTestCase
from boto.cloudformation.connection import CloudFormationConnection
from boto.exception import BotoServerError
from boto.compat import json
class TestCloudFormationValidateTemplate(CloudFormationConnectionBase):

    def default_body(self):
        return b'\n            <ValidateTemplateResponse xmlns="http://cloudformation.amazonaws.com/doc/2010-05-15/">\n              <ValidateTemplateResult>\n                <Description>My Description.</Description>\n                <Parameters>\n                  <member>\n                    <NoEcho>false</NoEcho>\n                    <ParameterKey>InstanceType</ParameterKey>\n                    <Description>Type of instance to launch</Description>\n                    <DefaultValue>m1.small</DefaultValue>\n                  </member>\n                  <member>\n                    <NoEcho>false</NoEcho>\n                    <ParameterKey>KeyName</ParameterKey>\n                    <Description>EC2 KeyPair</Description>\n                  </member>\n                </Parameters>\n                <CapabilitiesReason>Reason</CapabilitiesReason>\n                <Capabilities>\n                  <member>CAPABILITY_IAM</member>\n                </Capabilities>\n              </ValidateTemplateResult>\n              <ResponseMetadata>\n                <RequestId>0be7b6e8-e4a0-11e0-a5bd-9f8d5a7dbc91</RequestId>\n              </ResponseMetadata>\n            </ValidateTemplateResponse>\n        '

    def test_validate_template(self):
        self.set_http_response(status_code=200)
        template = self.service_connection.validate_template(template_body=SAMPLE_TEMPLATE, template_url='http://url')
        self.assertEqual(template.description, 'My Description.')
        self.assertEqual(len(template.template_parameters), 2)
        param1, param2 = template.template_parameters
        self.assertEqual(param1.default_value, 'm1.small')
        self.assertEqual(param1.description, 'Type of instance to launch')
        self.assertEqual(param1.no_echo, True)
        self.assertEqual(param1.parameter_key, 'InstanceType')
        self.assertEqual(param2.default_value, None)
        self.assertEqual(param2.description, 'EC2 KeyPair')
        self.assertEqual(param2.no_echo, True)
        self.assertEqual(param2.parameter_key, 'KeyName')
        self.assertEqual(template.capabilities_reason, 'Reason')
        self.assertEqual(len(template.capabilities), 1)
        self.assertEqual(template.capabilities[0].value, 'CAPABILITY_IAM')
        self.assert_request_parameters({'Action': 'ValidateTemplate', 'TemplateBody': SAMPLE_TEMPLATE, 'TemplateURL': 'http://url', 'Version': '2010-05-15'})