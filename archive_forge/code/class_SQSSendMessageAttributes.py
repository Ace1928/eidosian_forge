from tests.unit import AWSMockServiceTestCase, MockServiceWithConfigTestCase
from tests.compat import mock
from boto.sqs.connection import SQSConnection
from boto.sqs.regioninfo import SQSRegionInfo
from boto.sqs.message import RawMessage
from boto.sqs.queue import Queue
from boto.connection import AWSQueryConnection
from nose.plugins.attrib import attr
class SQSSendMessageAttributes(AWSMockServiceTestCase):
    connection_class = SQSConnection

    def default_body(self):
        return '<SendMessageResponse>\n    <SendMessageResult>\n        <MD5OfMessageBody>\n            fafb00f5732ab283681e124bf8747ed1\n        </MD5OfMessageBody>\n        <MD5OfMessageAttributes>\n        3ae8f24a165a8cedc005670c81a27295\n        </MD5OfMessageAttributes>\n        <MessageId>\n            5fea7756-0ea4-451a-a703-a558b933e274\n        </MessageId>\n    </SendMessageResult>\n    <ResponseMetadata>\n        <RequestId>\n            27daac76-34dd-47df-bd01-1f6e873584a0\n        </RequestId>\n    </ResponseMetadata>\n</SendMessageResponse>\n'

    @attr(sqs=True)
    def test_send_message_attributes(self):
        self.set_http_response(status_code=200)
        queue = Queue(url='http://sqs.us-east-1.amazonaws.com/123456789012/testQueue/', message_class=RawMessage)
        self.service_connection.send_message(queue, 'Test message', message_attributes={'name1': {'data_type': 'String', 'string_value': 'Bob'}, 'name2': {'data_type': 'Number', 'string_value': '1'}})
        self.assert_request_parameters({'Action': 'SendMessage', 'MessageAttribute.1.Name': 'name1', 'MessageAttribute.1.Value.DataType': 'String', 'MessageAttribute.1.Value.StringValue': 'Bob', 'MessageAttribute.2.Name': 'name2', 'MessageAttribute.2.Value.DataType': 'Number', 'MessageAttribute.2.Value.StringValue': '1', 'MessageBody': 'Test message', 'Version': '2012-11-05'})