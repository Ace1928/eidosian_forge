import unittest
from datetime import datetime
from mock import Mock
from tests.unit import AWSMockServiceTestCase
from boto.cloudformation.connection import CloudFormationConnection
from boto.exception import BotoServerError
from boto.compat import json
class TestCloudFormationEstimateTemplateCost(CloudFormationConnectionBase):

    def default_body(self):
        return b'\n            {\n                "EstimateTemplateCostResponse": {\n                    "EstimateTemplateCostResult": {\n                        "Url": "http://calculator.s3.amazonaws.com/calc5.html?key=cf-2e351785-e821-450c-9d58-625e1e1ebfb6"\n                    }\n                }\n            }\n        '

    def test_estimate_template_cost(self):
        self.set_http_response(status_code=200)
        api_response = self.service_connection.estimate_template_cost(template_body='{}')
        self.assertEqual(api_response, 'http://calculator.s3.amazonaws.com/calc5.html?key=cf-2e351785-e821-450c-9d58-625e1e1ebfb6')
        self.assert_request_parameters({'Action': 'EstimateTemplateCost', 'ContentType': 'JSON', 'TemplateBody': '{}', 'Version': '2010-05-15'})