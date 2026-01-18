from unittest import mock
import fixtures
from oslotest import base as test_base
from oslo_middleware import correlation_id
class CorrelationIdTest(test_base.BaseTestCase):

    def setUp(self):
        super(CorrelationIdTest, self).setUp()

    def test_process_request(self):
        app = mock.Mock()
        req = mock.Mock()
        req.headers = {}
        mock_uuid4 = mock.Mock()
        mock_uuid4.return_value = 'fake_uuid'
        self.useFixture(fixtures.MockPatch('uuid.uuid4', mock_uuid4))
        middleware = correlation_id.CorrelationId(app)
        middleware(req)
        self.assertEqual('fake_uuid', req.headers.get('X_CORRELATION_ID'))

    def test_process_request_should_not_regenerate_correlation_id(self):
        app = mock.Mock()
        req = mock.Mock()
        req.headers = {'X_CORRELATION_ID': 'correlation_id'}
        middleware = correlation_id.CorrelationId(app)
        middleware(req)
        self.assertEqual('correlation_id', req.headers.get('X_CORRELATION_ID'))