from tests.unit import AWSMockServiceTestCase, MockServiceWithConfigTestCase
from tests.compat import mock
from boto.sqs.connection import SQSConnection
from boto.sqs.regioninfo import SQSRegionInfo
from boto.sqs.message import RawMessage
from boto.sqs.queue import Queue
from boto.connection import AWSQueryConnection
from nose.plugins.attrib import attr
class SQSSendBatchMessageAttributes(AWSMockServiceTestCase):
    connection_class = SQSConnection

    def default_body(self):
        return '<SendMessageBatchResponse>\n<SendMessageBatchResult>\n    <SendMessageBatchResultEntry>\n        <Id>test_msg_001</Id>\n        <MessageId>0a5231c7-8bff-4955-be2e-8dc7c50a25fa</MessageId>\n        <MD5OfMessageBody>0e024d309850c78cba5eabbeff7cae71</MD5OfMessageBody>\n    </SendMessageBatchResultEntry>\n    <SendMessageBatchResultEntry>\n        <Id>test_msg_002</Id>\n        <MessageId>15ee1ed3-87e7-40c1-bdaa-2e49968ea7e9</MessageId>\n        <MD5OfMessageBody>7fb8146a82f95e0af155278f406862c2</MD5OfMessageBody>\n        <MD5OfMessageAttributes>295c5fa15a51aae6884d1d7c1d99ca50</MD5OfMessageAttributes>\n    </SendMessageBatchResultEntry>\n</SendMessageBatchResult>\n<ResponseMetadata>\n    <RequestId>ca1ad5d0-8271-408b-8d0f-1351bf547e74</RequestId>\n</ResponseMetadata>\n</SendMessageBatchResponse>\n'

    @attr(sqs=True)
    def test_send_message_attributes(self):
        self.set_http_response(status_code=200)
        queue = Queue(url='http://sqs.us-east-1.amazonaws.com/123456789012/testQueue/', message_class=RawMessage)
        message1 = (1, 'Message 1', 0, {'name1': {'data_type': 'String', 'string_value': 'foo'}})
        message2 = (2, 'Message 2', 0, {'name2': {'data_type': 'Number', 'string_value': '1'}})
        self.service_connection.send_message_batch(queue, (message1, message2))
        self.assert_request_parameters({'Action': 'SendMessageBatch', 'SendMessageBatchRequestEntry.1.DelaySeconds': 0, 'SendMessageBatchRequestEntry.1.Id': 1, 'SendMessageBatchRequestEntry.1.MessageAttribute.1.Name': 'name1', 'SendMessageBatchRequestEntry.1.MessageAttribute.1.Value.DataType': 'String', 'SendMessageBatchRequestEntry.1.MessageAttribute.1.Value.StringValue': 'foo', 'SendMessageBatchRequestEntry.1.MessageBody': 'Message 1', 'SendMessageBatchRequestEntry.2.DelaySeconds': 0, 'SendMessageBatchRequestEntry.2.Id': 2, 'SendMessageBatchRequestEntry.2.MessageAttribute.1.Name': 'name2', 'SendMessageBatchRequestEntry.2.MessageAttribute.1.Value.DataType': 'Number', 'SendMessageBatchRequestEntry.2.MessageAttribute.1.Value.StringValue': '1', 'SendMessageBatchRequestEntry.2.MessageBody': 'Message 2', 'Version': '2012-11-05'})