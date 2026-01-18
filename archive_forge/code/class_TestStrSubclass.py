from unittest import TestCase
import simplejson
from simplejson.compat import text_type
class TestStrSubclass(TestCase):

    def test_dump_load(self):
        for s in ['', '"hello"', 'text', u'\\']:
            self.assertEqual(s, simplejson.loads(simplejson.dumps(WonkyTextSubclass(s))))
            self.assertEqual(s, simplejson.loads(simplejson.dumps(WonkyTextSubclass(s), ensure_ascii=False)))