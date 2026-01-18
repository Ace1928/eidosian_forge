from oslo_log import fixture
from oslo_log import log as logging
from oslotest import base as test_base
class TestLoggingFixture(test_base.BaseTestCase):

    def setUp(self):
        super(TestLoggingFixture, self).setUp()
        self.log = logging.getLogger(__name__)

    def test_logging_handle_error(self):
        self.log.error('pid of first child is %(foo)s', 1)
        self.useFixture(fixture.get_logging_handle_error_fixture())
        self.assertRaises(TypeError, self.log.error, 'pid of first child is %(foo)s', 1)