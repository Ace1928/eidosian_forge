import unittest
import logging
from os_ken.ofproto.ofproto_v1_0_parser import *
from os_ken.ofproto.nx_actions import *
from os_ken.ofproto import ofproto_v1_0_parser
from os_ken.lib import addrconv
class TestNXActionBundle(unittest.TestCase):
    """ Test case for ofproto_v1_0_parser.NXActionBundle
    """
    type_ = {'buf': b'\xff\xff', 'val': ofproto.OFPAT_VENDOR}
    len_ = {'buf': b'\x00(', 'val': ofproto.NX_ACTION_BUNDLE_SIZE + 8}
    vendor = {'buf': b'\x00\x00# ', 'val': ofproto_common.NX_EXPERIMENTER_ID}
    subtype = {'buf': b'\x00\x0c', 'val': ofproto.NXAST_BUNDLE}
    algorithm = {'buf': b'Q\xa7', 'val': 20903}
    fields = {'buf': b'\xf8\xef', 'val': 63727}
    basis = {'buf': b'\xfdo', 'val': 64879}
    slave_type = {'buf': b'|Q\x0f\xe0', 'val': 2085687264}
    n_slaves = {'buf': b'\x00\x02', 'val': 2}
    ofs_nbits = {'buf': b'\x00\x00', 'val': 0}
    dst = {'buf': b'\x00\x00\x00\x00', 'val': 0}
    zfill = b'\x00' * 4
    slaves_buf = (b'\x00\x01', b'\x00\x02')
    slaves_val = (1, 2)
    _len = len_['val'] + len(slaves_val) * 2
    _len += _len % 8
    buf = type_['buf'] + len_['buf'] + vendor['buf'] + subtype['buf'] + algorithm['buf'] + fields['buf'] + basis['buf'] + slave_type['buf'] + n_slaves['buf'] + ofs_nbits['buf'] + dst['buf'] + zfill + slaves_buf[0] + slaves_buf[1]
    c = NXActionBundle(algorithm['val'], fields['val'], basis['val'], slave_type['val'], n_slaves['val'], ofs_nbits['val'], dst['val'], slaves_val)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_init(self):
        self.assertEqual(self.subtype['val'], self.c.subtype)
        self.assertEqual(self.algorithm['val'], self.c.algorithm)
        self.assertEqual(self.fields['val'], self.c.fields)
        self.assertEqual(self.basis['val'], self.c.basis)
        self.assertEqual(self.slave_type['val'], self.c.slave_type)
        self.assertEqual(self.n_slaves['val'], self.c.n_slaves)
        self.assertEqual(self.ofs_nbits['val'], self.c.ofs_nbits)
        self.assertEqual(self.dst['val'], self.c.dst)
        slaves = self.c.slaves
        self.assertEqual(self.slaves_val[0], slaves[0])
        self.assertEqual(self.slaves_val[1], slaves[1])

    def test_parser(self):
        res = OFPActionVendor.parser(self.buf, 0)
        self.assertEqual(self.type_['val'], res.type)
        self.assertEqual(self.len_['val'], res.len)
        self.assertEqual(self.subtype['val'], res.subtype)
        self.assertEqual(self.algorithm['val'], res.algorithm)
        self.assertEqual(self.fields['val'], res.fields)
        self.assertEqual(self.basis['val'], res.basis)
        self.assertEqual(self.slave_type['val'], res.slave_type)
        self.assertEqual(self.n_slaves['val'], res.n_slaves)
        self.assertEqual(self.ofs_nbits['val'], res.ofs_nbits)
        self.assertEqual(self.dst['val'], res.dst)
        slaves = res.slaves
        self.assertEqual(self.slaves_val[0], slaves[0])
        self.assertEqual(self.slaves_val[1], slaves[1])

    def test_serialize(self):
        buf = bytearray()
        self.c.serialize(buf, 0)
        fmt = '!' + ofproto.NX_ACTION_BUNDLE_PACK_STR.replace('!', '') + 'HH4x'
        res = struct.unpack(fmt, bytes(buf))
        self.assertEqual(self.type_['val'], res[0])
        self.assertEqual(self.len_['val'], res[1])
        self.assertEqual(self.vendor['val'], res[2])
        self.assertEqual(self.subtype['val'], res[3])
        self.assertEqual(self.algorithm['val'], res[4])
        self.assertEqual(self.fields['val'], res[5])
        self.assertEqual(self.basis['val'], res[6])
        self.assertEqual(self.slave_type['val'], res[7])
        self.assertEqual(self.n_slaves['val'], res[8])
        self.assertEqual(self.ofs_nbits['val'], res[9])
        self.assertEqual(self.dst['val'], res[10])