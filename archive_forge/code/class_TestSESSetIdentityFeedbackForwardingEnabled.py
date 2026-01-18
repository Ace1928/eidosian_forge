from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.jsonresponse import ListElement
from boto.ses.connection import SESConnection
class TestSESSetIdentityFeedbackForwardingEnabled(AWSMockServiceTestCase):
    connection_class = SESConnection

    def setUp(self):
        super(TestSESSetIdentityFeedbackForwardingEnabled, self).setUp()

    def default_body(self):
        return b'<SetIdentityFeedbackForwardingEnabledResponse         xmlns="http://ses.amazonaws.com/doc/2010-12-01/">\n          <SetIdentityFeedbackForwardingEnabledResult/>\n          <ResponseMetadata>\n            <RequestId>299f4af4-b72a-11e1-901f-1fbd90e8104f</RequestId>\n          </ResponseMetadata>\n        </SetIdentityFeedbackForwardingEnabledResponse>'

    def test_ses_set_identity_feedback_forwarding_enabled_true(self):
        self.set_http_response(status_code=200)
        response = self.service_connection.set_identity_feedback_forwarding_enabled(identity='user@example.com', forwarding_enabled=True)
        response = response['SetIdentityFeedbackForwardingEnabledResponse']
        result = response['SetIdentityFeedbackForwardingEnabledResult']
        self.assertEqual(2, len(response))
        self.assertEqual(0, len(result))

    def test_ses_set_identity_notification_topic_enabled_false(self):
        self.set_http_response(status_code=200)
        response = self.service_connection.set_identity_feedback_forwarding_enabled(identity='user@example.com', forwarding_enabled=False)
        response = response['SetIdentityFeedbackForwardingEnabledResponse']
        result = response['SetIdentityFeedbackForwardingEnabledResult']
        self.assertEqual(2, len(response))
        self.assertEqual(0, len(result))