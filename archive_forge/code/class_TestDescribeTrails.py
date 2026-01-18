import json
from boto.cloudtrail.layer1 import CloudTrailConnection
from tests.unit import AWSMockServiceTestCase
class TestDescribeTrails(AWSMockServiceTestCase):
    connection_class = CloudTrailConnection

    def default_body(self):
        return b'\n            {"trailList":\n                [\n                    {\n                        "IncludeGlobalServiceEvents": false,\n                        "Name": "test",\n                        "SnsTopicName": "cloudtrail-1",\n                        "S3BucketName": "cloudtrail-1"\n                    }\n                ]\n            }'

    def test_describe(self):
        self.set_http_response(status_code=200)
        api_response = self.service_connection.describe_trails()
        self.assertEqual(1, len(api_response['trailList']))
        self.assertEqual('test', api_response['trailList'][0]['Name'])
        self.assert_request_parameters({})
        target = self.actual_request.headers['X-Amz-Target']
        self.assertTrue('DescribeTrails' in target)

    def test_describe_name_list(self):
        self.set_http_response(status_code=200)
        api_response = self.service_connection.describe_trails(trail_name_list=['test'])
        self.assertEqual(1, len(api_response['trailList']))
        self.assertEqual('test', api_response['trailList'][0]['Name'])
        self.assertEqual(json.dumps({'trailNameList': ['test']}), self.actual_request.body.decode('utf-8'))
        target = self.actual_request.headers['X-Amz-Target']
        self.assertTrue('DescribeTrails' in target)