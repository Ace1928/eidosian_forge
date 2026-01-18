import uuid
import fixtures
from oslo_config import fixture as config_fixture
from oslo_log import log
from oslo_serialization import jsonutils
import keystone.conf
from keystone import exception
from keystone.server.flask.request_processing.middleware import auth_context
from keystone.tests import unit
class TestSecurityErrorTranslation(unit.BaseTestCase):
    """Test i18n for SecurityError exceptions."""

    def setUp(self):
        super(TestSecurityErrorTranslation, self).setUp()
        self.config_fixture = self.useFixture(config_fixture.Config(CONF))
        self.config_fixture.config(insecure_debug=False)
        self.warning_log = self.useFixture(fixtures.FakeLogger(level=log.WARN))
        exception._FATAL_EXCEPTION_FORMAT_ERRORS = False
        self.addCleanup(setattr, exception, '_FATAL_EXCEPTION_FORMAT_ERRORS', True)

    class CustomSecurityError(exception.SecurityError):
        message_format = 'We had a failure in the %(place)r'

    class CustomError(exception.Error):
        message_format = 'We had a failure in the %(place)r'

    def test_nested_translation_of_SecurityErrors(self):
        e = self.CustomSecurityError(place='code')
        'Admiral found this in the log: %s' % e
        self.assertNotIn('programmer error', self.warning_log.output)

    def test_that_regular_Errors_can_be_deep_copied(self):
        e = self.CustomError(place='code')
        'Admiral found this in the log: %s' % e
        self.assertNotIn('programmer error', self.warning_log.output)