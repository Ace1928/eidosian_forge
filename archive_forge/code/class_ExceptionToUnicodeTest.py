from unittest import mock
from oslo_i18n import fixture as oslo_i18n_fixture
from oslotest import base as test_base
from oslo_utils import encodeutils
class ExceptionToUnicodeTest(test_base.BaseTestCase):

    def test_str_exception(self):

        class StrException(Exception):

            def __init__(self, value):
                Exception.__init__(self)
                self.value = value

            def __str__(self):
                return self.value
        exc = StrException(b'bytes ascii')
        self.assertEqual(encodeutils.exception_to_unicode(exc), 'bytes ascii')
        exc = StrException(b'utf-8 \xc3\xa9\xe2\x82\xac')
        self.assertEqual(encodeutils.exception_to_unicode(exc), 'utf-8 é€')
        with mock.patch.object(encodeutils, '_getfilesystemencoding', return_value='ascii'):
            exc = StrException(b'rawbytes \x80\xff')
            self.assertEqual(encodeutils.exception_to_unicode(exc), 'rawbytes \x80ÿ')
        exc = StrException('unicode ascii')
        self.assertEqual(encodeutils.exception_to_unicode(exc), 'unicode ascii')
        exc = StrException('unicode é€')
        self.assertEqual(encodeutils.exception_to_unicode(exc), 'unicode é€')
        with mock.patch.object(encodeutils, '_getfilesystemencoding', return_value='koi8_r'):
            exc = StrException(b'\xf2\xd5\xd3\xd3\xcb\xc9\xca')
            self.assertEqual(encodeutils.exception_to_unicode(exc), 'Русский')

    def test_oslo_i18n_message(self):
        exc = oslo_i18n_fixture.Translation().lazy('test')
        self.assertEqual(encodeutils.exception_to_unicode(exc), 'test')