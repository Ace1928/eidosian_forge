from functools import reduce
import unittest
import testscenarios
from os_ken.ofproto import ofproto_v1_5
from os_ken.ofproto import ofproto_v1_5_parser
class Test_Parser_OFPStats(testscenarios.WithScenarios, unittest.TestCase):
    scenarios = [(case['name'], case) for case in _list_test_cases()]
    _ofp = {ofproto_v1_5_parser: ofproto_v1_5}

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_parser(self):
        self._test(name=self.name, ofpp=self.ofpp, d=self.d)

    def _test(self, name, ofpp, d):
        stats = ofpp.OFPStats(**d)
        b = bytearray()
        stats.serialize(b, 0)
        stats2 = stats.parser(bytes(b), 0)
        for k, v in d.items():
            self.assertTrue(k in stats)
            self.assertTrue(k in stats2)
            self.assertEqual(stats[k], v)
            self.assertEqual(stats2[k], v)
        for k, v in stats.iteritems():
            self.assertTrue(k in d)
            self.assertEqual(d[k], v)
        for k, v in stats2.iteritems():
            self.assertTrue(k in d)
            self.assertEqual(d[k], v)