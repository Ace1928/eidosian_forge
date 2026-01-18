from taskflow import test
from taskflow.utils import misc
class BinaryDecodeTest(test.TestCase):

    def _check(self, data, expected_result):
        result = misc.binary_decode(data)
        self.assertIsInstance(result, str)
        self.assertEqual(expected_result, result)

    def test_simple_text(self):
        data = u'hello'
        self._check(data, data)

    def test_unicode_text(self):
        data = u'привет'
        self._check(data, data)

    def test_simple_binary(self):
        self._check(_bytes('hello'), u'hello')

    def test_unicode_binary(self):
        self._check(_bytes('привет'), u'привет')

    def test_unicode_other_encoding(self):
        data = u'mañana'.encode('latin-1')
        result = misc.binary_decode(data, 'latin-1')
        self.assertIsInstance(result, str)
        self.assertEqual(u'mañana', result)