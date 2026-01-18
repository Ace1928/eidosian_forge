from tests.unit import AWSMockServiceTestCase, MockServiceWithConfigTestCase
from tests.compat import mock
from boto.sqs.connection import SQSConnection
from boto.sqs.regioninfo import SQSRegionInfo
from boto.sqs.message import RawMessage
from boto.sqs.queue import Queue
from boto.connection import AWSQueryConnection
from nose.plugins.attrib import attr
class SQSMessageAttributesParsing(AWSMockServiceTestCase):
    connection_class = SQSConnection

    def default_body(self):
        return '<?xml version="1.0"?>\n<ReceiveMessageResponse xmlns="http://queue.amazonaws.com/doc/2012-11-05/">\n    <ReceiveMessageResult>\n        <Message>\n            <Body>This is a test</Body>\n            <ReceiptHandle>+eXJYhj5rDql5hp2VwGkXvQVsefdjAlsQe5EGS57gyORPB48KwP1d/3Rfy4DrQXt+MgfRPHUCUH36xL9+Ol/UWD/ylKrrWhiXSY0Ip4EsI8jJNTo/aneEjKE/iZnz/nL8MFP5FmMj8PbDAy5dgvAqsdvX1rm8Ynn0bGnQLJGfH93cLXT65p6Z/FDyjeBN0M+9SWtTcuxOIcMdU8NsoFIwm/6mLWgWAV46OhlYujzvyopCvVwsj+Y8jLEpdSSvTQHNlQEaaY/V511DqAvUwru2p0ZbW7ZzcbhUTn6hHkUROo=</ReceiptHandle>\n            <MD5OfBody>ce114e4501d2f4e2dcea3e17b546f339</MD5OfBody>\n            <MessageAttribute>\n                <Name>Count</Name>\n                <Value>\n                    <DataType>Number</DataType>\n                    <StringValue>1</StringValue>\n                </Value>\n            </MessageAttribute>\n            <MessageAttribute>\n                <Name>Foo</Name>\n                <Value>\n                    <DataType>String</DataType>\n                    <StringValue>Bar</StringValue>\n                </Value>\n            </MessageAttribute>\n            <MessageId>7049431b-e5f6-430b-93c4-ded53864d02b</MessageId>\n            <MD5OfMessageAttributes>324758f82d026ac6ec5b31a3b192d1e3</MD5OfMessageAttributes>\n        </Message>\n    </ReceiveMessageResult>\n    <ResponseMetadata>\n        <RequestId>73f978f2-400b-5460-8d38-3316e39e79c6</RequestId>\n    </ResponseMetadata>\n</ReceiveMessageResponse>'

    @attr(sqs=True)
    def test_message_attribute_response(self):
        self.set_http_response(status_code=200)
        queue = Queue(url='http://sqs.us-east-1.amazonaws.com/123456789012/testQueue/', message_class=RawMessage)
        message = self.service_connection.receive_message(queue)[0]
        self.assertEqual(message.get_body(), 'This is a test')
        self.assertEqual(message.id, '7049431b-e5f6-430b-93c4-ded53864d02b')
        self.assertEqual(message.md5, 'ce114e4501d2f4e2dcea3e17b546f339')
        self.assertEqual(message.md5_message_attributes, '324758f82d026ac6ec5b31a3b192d1e3')
        mattributes = message.message_attributes
        self.assertEqual(len(mattributes.keys()), 2)
        self.assertEqual(mattributes['Count']['data_type'], 'Number')
        self.assertEqual(mattributes['Foo']['string_value'], 'Bar')