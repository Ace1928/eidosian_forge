from tests.unit import AWSMockServiceTestCase, MockServiceWithConfigTestCase
from tests.compat import mock
from boto.sqs.connection import SQSConnection
from boto.sqs.regioninfo import SQSRegionInfo
from boto.sqs.message import RawMessage
from boto.sqs.queue import Queue
from boto.connection import AWSQueryConnection
from nose.plugins.attrib import attr
class SQSAuthParams(AWSMockServiceTestCase):
    connection_class = SQSConnection

    def setUp(self):
        super(SQSAuthParams, self).setUp()

    def default_body(self):
        return '<?xml version="1.0"?>\n            <CreateQueueResponse>\n              <CreateQueueResult>\n                <QueueUrl>\n                  https://queue.amazonaws.com/599169622985/myqueue1\n                </QueueUrl>\n              </CreateQueueResult>\n              <ResponseMetadata>\n                <RequestId>54d4c94d-2307-54a8-bb27-806a682a5abd</RequestId>\n              </ResponseMetadata>\n            </CreateQueueResponse>'

    @attr(sqs=True)
    def test_auth_service_name_override(self):
        self.set_http_response(status_code=200)
        self.service_connection.auth_service_name = 'service_override'
        self.service_connection.create_queue('my_queue')
        self.assertIn('us-east-1/service_override/aws4_request', self.actual_request.headers['Authorization'])

    @attr(sqs=True)
    def test_class_attribute_can_set_service_name(self):
        self.set_http_response(status_code=200)
        self.assertEqual(self.service_connection.AuthServiceName, 'sqs')
        self.service_connection.create_queue('my_queue')
        self.assertIn('us-east-1/sqs/aws4_request', self.actual_request.headers['Authorization'])

    @attr(sqs=True)
    def test_auth_region_name_is_automatically_updated(self):
        region = SQSRegionInfo(name='us-west-2', endpoint='us-west-2.queue.amazonaws.com')
        self.service_connection = SQSConnection(https_connection_factory=self.https_connection_factory, aws_access_key_id='aws_access_key_id', aws_secret_access_key='aws_secret_access_key', region=region)
        self.initialize_service_connection()
        self.set_http_response(status_code=200)
        self.service_connection.create_queue('my_queue')
        self.assertIn('us-west-2/sqs/aws4_request', self.actual_request.headers['Authorization'])

    @attr(sqs=True)
    def test_set_get_auth_service_and_region_names(self):
        self.service_connection.auth_service_name = 'service_name'
        self.service_connection.auth_region_name = 'region_name'
        self.assertEqual(self.service_connection.auth_service_name, 'service_name')
        self.assertEqual(self.service_connection.auth_region_name, 'region_name')

    @attr(sqs=True)
    def test_get_queue_with_owner_account_id_returns_queue(self):
        self.set_http_response(status_code=200)
        self.service_connection.create_queue('my_queue')
        self.service_connection.get_queue('my_queue', '599169622985')
        assert 'QueueOwnerAWSAccountId' in self.actual_request.params.keys()
        self.assertEquals(self.actual_request.params['QueueOwnerAWSAccountId'], '599169622985')