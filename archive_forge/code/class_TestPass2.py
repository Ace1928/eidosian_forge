from unittest import TestCase
import simplejson as json
class TestPass2(TestCase):

    def test_parse(self):
        res = json.loads(JSON)
        out = json.dumps(res)
        self.assertEqual(res, json.loads(out))