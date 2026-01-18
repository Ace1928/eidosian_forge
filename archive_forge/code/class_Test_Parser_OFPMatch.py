import functools
import itertools
import unittest
import testscenarios
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_v1_3
from os_ken.ofproto import ofproto_v1_4
from os_ken.ofproto import ofproto_v1_5
from os_ken.ofproto import ofproto_v1_2_parser
from os_ken.ofproto import ofproto_v1_3_parser
from os_ken.ofproto import ofproto_v1_4_parser
from os_ken.ofproto import ofproto_v1_5_parser
class Test_Parser_OFPMatch(testscenarios.WithScenarios, unittest.TestCase):
    scenarios = [(case['name'], case) for case in _list_test_cases()]
    _ofp = {ofproto_v1_2_parser: ofproto_v1_2, ofproto_v1_3_parser: ofproto_v1_3, ofproto_v1_4_parser: ofproto_v1_4, ofproto_v1_5_parser: ofproto_v1_5}

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_parser(self):
        self._test(name=self.name, ofpp=self.ofpp, d=self.d, domask=self.domask)

    def _test(self, name, ofpp, d, domask):
        if domask:
            d = dict((self._ofp[ofpp].oxm_normalize_user(k, uv) for k, uv in d.items()))
        match = ofpp.OFPMatch(**d)
        b = bytearray()
        match.serialize(b, 0)
        match2 = match.parser(bytes(b), 0)
        for k, v in d.items():
            self.assertTrue(k in match)
            self.assertTrue(k in match2)
            self.assertEqual(match[k], v)
            self.assertEqual(match2[k], v)
        for k, v in match.iteritems():
            self.assertTrue(k in d)
            self.assertEqual(d[k], v)
        for k, v in match2.iteritems():
            self.assertTrue(k in d)
            self.assertEqual(d[k], v)