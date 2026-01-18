from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.jsonresponse import ListElement
from boto.ses.connection import SESConnection
class TestSESSetIdentityNotificationTopic(AWSMockServiceTestCase):
    connection_class = SESConnection

    def setUp(self):
        super(TestSESSetIdentityNotificationTopic, self).setUp()

    def default_body(self):
        return b'<SetIdentityNotificationTopicResponse         xmlns="http://ses.amazonaws.com/doc/2010-12-01/">\n           <SetIdentityNotificationTopicResult/>\n           <ResponseMetadata>\n             <RequestId>299f4af4-b72a-11e1-901f-1fbd90e8104f</RequestId>\n           </ResponseMetadata>\n         </SetIdentityNotificationTopicResponse>'

    def test_ses_set_identity_notification_topic_bounce(self):
        self.set_http_response(status_code=200)
        response = self.service_connection.set_identity_notification_topic(identity='user@example.com', notification_type='Bounce', sns_topic='arn:aws:sns:us-east-1:123456789012:example')
        response = response['SetIdentityNotificationTopicResponse']
        result = response['SetIdentityNotificationTopicResult']
        self.assertEqual(2, len(response))
        self.assertEqual(0, len(result))

    def test_ses_set_identity_notification_topic_complaint(self):
        self.set_http_response(status_code=200)
        response = self.service_connection.set_identity_notification_topic(identity='user@example.com', notification_type='Complaint', sns_topic='arn:aws:sns:us-east-1:123456789012:example')
        response = response['SetIdentityNotificationTopicResponse']
        result = response['SetIdentityNotificationTopicResult']
        self.assertEqual(2, len(response))
        self.assertEqual(0, len(result))