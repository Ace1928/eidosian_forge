from unittest import mock
from oslo_i18n import fixture as oslo_i18n_fixture
from oslotest import base as test_base
from oslo_utils import encodeutils
class EncodeUtilsTest(test_base.BaseTestCase):

    def test_safe_decode(self):
        safe_decode = encodeutils.safe_decode
        self.assertRaises(TypeError, safe_decode, True)
        self.assertEqual('niño', safe_decode('niÃ±o'.encode('latin-1'), incoming='utf-8'))
        self.assertEqual('strange', safe_decode('\x80strange'.encode('latin-1'), errors='ignore'))
        self.assertEqual('À', safe_decode('À'.encode('latin-1'), incoming='iso-8859-1'))
        self.assertEqual('niño', safe_decode('niÃ±o'.encode('latin-1'), incoming='ascii'))
        self.assertEqual('foo', safe_decode(b'foo'))

    def test_safe_encode_none_instead_of_text(self):
        self.assertRaises(TypeError, encodeutils.safe_encode, None)

    def test_safe_encode_bool_instead_of_text(self):
        self.assertRaises(TypeError, encodeutils.safe_encode, True)

    def test_safe_encode_int_instead_of_text(self):
        self.assertRaises(TypeError, encodeutils.safe_encode, 1)

    def test_safe_encode_list_instead_of_text(self):
        self.assertRaises(TypeError, encodeutils.safe_encode, [])

    def test_safe_encode_dict_instead_of_text(self):
        self.assertRaises(TypeError, encodeutils.safe_encode, {})

    def test_safe_encode_tuple_instead_of_text(self):
        self.assertRaises(TypeError, encodeutils.safe_encode, ('foo', 'bar'))

    def test_safe_encode_force_incoming_utf8_to_ascii(self):
        self.assertEqual('niÃ±o'.encode('latin-1'), encodeutils.safe_encode('niÃ±o'.encode('latin-1'), incoming='ascii'))

    def test_safe_encode_same_encoding_different_cases(self):
        with mock.patch.object(encodeutils, 'safe_decode', mock.Mock()):
            utf8 = encodeutils.safe_encode('fooñbar', encoding='utf-8')
            self.assertEqual(encodeutils.safe_encode(utf8, 'UTF-8', 'utf-8'), encodeutils.safe_encode(utf8, 'utf-8', 'UTF-8'))
            self.assertEqual(encodeutils.safe_encode(utf8, 'UTF-8', 'utf-8'), encodeutils.safe_encode(utf8, 'utf-8', 'utf-8'))
            encodeutils.safe_decode.assert_has_calls([])

    def test_safe_encode_different_encodings(self):
        text = 'fooÃ±bar'
        result = encodeutils.safe_encode(text=text, incoming='utf-8', encoding='iso-8859-1')
        self.assertNotEqual(text, result)
        self.assertNotEqual('fooñbar'.encode('latin-1'), result)

    def test_to_utf8(self):
        self.assertEqual(encodeutils.to_utf8(b'a\xe9\xff'), b'a\xe9\xff')
        self.assertEqual(encodeutils.to_utf8('aéÿ€'), b'a\xc3\xa9\xc3\xbf\xe2\x82\xac')
        self.assertRaises(TypeError, encodeutils.to_utf8, 123)
        msg = oslo_i18n_fixture.Translation().lazy('test')
        self.assertEqual(encodeutils.to_utf8(msg), b'test')