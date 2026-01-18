import http.client as http
from oslo_utils import encodeutils
from glance.common import exception
from glance.tests import utils as test_utils
class GlanceExceptionTestCase(test_utils.BaseTestCase):

    def test_default_error_msg(self):

        class FakeGlanceException(exception.GlanceException):
            message = 'default message'
        exc = FakeGlanceException()
        self.assertEqual('default message', encodeutils.exception_to_unicode(exc))

    def test_specified_error_msg(self):
        msg = exception.GlanceException('test')
        self.assertIn('test', encodeutils.exception_to_unicode(msg))

    def test_default_error_msg_with_kwargs(self):

        class FakeGlanceException(exception.GlanceException):
            message = 'default message: %(code)s'
        exc = FakeGlanceException(code=int(http.INTERNAL_SERVER_ERROR))
        self.assertEqual('default message: 500', encodeutils.exception_to_unicode(exc))

    def test_specified_error_msg_with_kwargs(self):
        msg = exception.GlanceException('test: %(code)s', code=int(http.INTERNAL_SERVER_ERROR))
        self.assertIn('test: 500', encodeutils.exception_to_unicode(msg))

    def test_non_unicode_error_msg(self):
        exc = exception.GlanceException('test')
        self.assertIsInstance(encodeutils.exception_to_unicode(exc), str)