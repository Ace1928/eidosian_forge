from unittest import mock
import fixtures
from heat.common import exception
from heat.common.i18n import _
from heat.tests import common
class TestHeatException(common.HeatTestCase):

    def test_fatal_exception_error(self):
        self.useFixture(fixtures.MonkeyPatch('heat.common.exception._FATAL_EXCEPTION_FORMAT_ERRORS', True))
        self.assertRaises(KeyError, TestException)

    def test_format_string_error_message(self):
        message = 'This format %(message)s should work'
        err = exception.Error(message)
        self.assertEqual(message, str(err))