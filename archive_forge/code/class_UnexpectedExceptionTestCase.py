import uuid
import fixtures
from oslo_config import fixture as config_fixture
from oslo_log import log
from oslo_serialization import jsonutils
import keystone.conf
from keystone import exception
from keystone.server.flask.request_processing.middleware import auth_context
from keystone.tests import unit
class UnexpectedExceptionTestCase(ExceptionTestCase):
    """Test if internal info is exposed to the API user on UnexpectedError."""

    class SubClassExc(exception.UnexpectedError):
        debug_message_format = 'Debug Message: %(debug_info)s'

    def setUp(self):
        super(UnexpectedExceptionTestCase, self).setUp()
        self.exc_str = uuid.uuid4().hex
        self.config_fixture = self.useFixture(config_fixture.Config(CONF))

    def test_unexpected_error_no_debug(self):
        self.config_fixture.config(debug=False)
        e = exception.UnexpectedError(exception=self.exc_str)
        self.assertNotIn(self.exc_str, str(e))

    def test_unexpected_error_debug(self):
        self.config_fixture.config(debug=True, insecure_debug=True)
        e = exception.UnexpectedError(exception=self.exc_str)
        self.assertIn(self.exc_str, str(e))

    def test_unexpected_error_subclass_no_debug(self):
        self.config_fixture.config(debug=False)
        e = UnexpectedExceptionTestCase.SubClassExc(debug_info=self.exc_str)
        self.assertEqual(exception.UnexpectedError.message_format, str(e))

    def test_unexpected_error_subclass_debug(self):
        self.config_fixture.config(debug=True, insecure_debug=True)
        subclass = self.SubClassExc
        e = subclass(debug_info=self.exc_str)
        expected = subclass.debug_message_format % {'debug_info': self.exc_str}
        self.assertEqual('%s %s' % (expected, exception.SecurityError.amendment), str(e))

    def test_unexpected_error_custom_message_no_debug(self):
        self.config_fixture.config(debug=False)
        e = exception.UnexpectedError(self.exc_str)
        self.assertEqual(exception.UnexpectedError.message_format, str(e))

    def test_unexpected_error_custom_message_debug(self):
        self.config_fixture.config(debug=True, insecure_debug=True)
        e = exception.UnexpectedError(self.exc_str)
        self.assertEqual('%s %s' % (self.exc_str, exception.SecurityError.amendment), str(e))

    def test_unexpected_error_custom_message_exception_debug(self):
        self.config_fixture.config(debug=True, insecure_debug=True)
        orig_e = exception.NotFound(target=uuid.uuid4().hex)
        e = exception.UnexpectedError(orig_e)
        self.assertEqual('%s %s' % (str(orig_e), exception.SecurityError.amendment), str(e))

    def test_unexpected_error_custom_message_binary_debug(self):
        self.config_fixture.config(debug=True, insecure_debug=True)
        binary_msg = b'something'
        e = exception.UnexpectedError(binary_msg)
        self.assertEqual('%s %s' % (str(binary_msg), exception.SecurityError.amendment), str(e))