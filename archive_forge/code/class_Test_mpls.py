import unittest
import logging
import inspect
from os_ken.lib.packet import mpls
class Test_mpls(unittest.TestCase):
    label = 29
    exp = 6
    bsb = 1
    ttl = 64
    mp = mpls.mpls(label, exp, bsb, ttl)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_to_string(self):
        mpls_values = {'label': self.label, 'exp': self.exp, 'bsb': self.bsb, 'ttl': self.ttl}
        _mpls_str = ','.join(['%s=%s' % (k, repr(mpls_values[k])) for k, v in inspect.getmembers(self.mp) if k in mpls_values])
        mpls_str = '%s(%s)' % (mpls.mpls.__name__, _mpls_str)
        self.assertEqual(str(self.mp), mpls_str)
        self.assertEqual(repr(self.mp), mpls_str)

    def test_json(self):
        jsondict = self.mp.to_jsondict()
        mp = mpls.mpls.from_jsondict(jsondict['mpls'])
        self.assertEqual(str(self.mp), str(mp))

    def test_label_from_bin_true(self):
        mpls_label = 1048575
        is_bos = True
        buf = b'\xff\xff\xf1'
        mpls_label_out, is_bos_out = mpls.label_from_bin(buf)
        self.assertEqual(mpls_label, mpls_label_out)
        self.assertEqual(is_bos, is_bos_out)

    def test_label_from_bin_false(self):
        mpls_label = 1048575
        is_bos = False
        buf = b'\xff\xff\xf0'
        mpls_label_out, is_bos_out = mpls.label_from_bin(buf)
        self.assertEqual(mpls_label, mpls_label_out)
        self.assertEqual(is_bos, is_bos_out)

    def test_label_to_bin_true(self):
        mpls_label = 1048575
        is_bos = True
        label = b'\xff\xff\xf1'
        label_out = mpls.label_to_bin(mpls_label, is_bos)
        self.assertEqual(label, label_out)

    def test_label_to_bin_false(self):
        mpls_label = 1048575
        is_bos = False
        label = b'\xff\xff\xf0'
        label_out = mpls.label_to_bin(mpls_label, is_bos)
        self.assertEqual(label, label_out)