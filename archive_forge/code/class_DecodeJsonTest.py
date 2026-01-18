from taskflow import test
from taskflow.utils import misc
class DecodeJsonTest(test.TestCase):

    def test_it_works(self):
        self.assertEqual({'foo': 1}, misc.decode_json(_bytes('{"foo": 1}')))

    def test_it_works_with_unicode(self):
        data = _bytes('{"foo": "фуу"}')
        self.assertEqual({'foo': u'фуу'}, misc.decode_json(data))

    def test_handles_invalid_unicode(self):
        self.assertRaises(ValueError, misc.decode_json, '{"ñ": 1}'.encode('latin-1'))

    def test_handles_bad_json(self):
        self.assertRaises(ValueError, misc.decode_json, _bytes('{"foo":'))

    def test_handles_wrong_types(self):
        self.assertRaises(ValueError, misc.decode_json, _bytes('42'))