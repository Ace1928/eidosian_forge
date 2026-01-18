import unittest
import logging
import inspect
import struct
from os_ken.lib import addrconv
from os_ken.lib.packet import cfm
class Test_loopback_reply(unittest.TestCase):

    def setUp(self):
        self.md_lv = 1
        self.version = 1
        self.opcode = cfm.CFM_LOOPBACK_REPLY
        self.flags = 0
        self.first_tlv_offset = cfm.loopback_reply._TLV_OFFSET
        self.transaction_id = 12345
        self.tlvs = []
        self.end_tlv = 0
        self.ins = cfm.loopback_reply(self.md_lv, self.version, self.transaction_id, self.tlvs)
        self.form = '!4BIB'
        self.buf = struct.pack(self.form, self.md_lv << 5 | self.version, self.opcode, self.flags, self.first_tlv_offset, self.transaction_id, self.end_tlv)

    def tearDown(self):
        pass

    def test_init(self):
        self.assertEqual(self.md_lv, self.ins.md_lv)
        self.assertEqual(self.version, self.ins.version)
        self.assertEqual(self.transaction_id, self.ins.transaction_id)
        self.assertEqual(self.tlvs, self.ins.tlvs)

    def test_parser(self):
        _res = cfm.loopback_reply.parser(self.buf)
        if type(_res) is tuple:
            res = _res[0]
        else:
            res = _res
        self.assertEqual(self.md_lv, res.md_lv)
        self.assertEqual(self.version, res.version)
        self.assertEqual(self.transaction_id, res.transaction_id)
        self.assertEqual(self.tlvs, res.tlvs)

    def test_serialize(self):
        buf = self.ins.serialize()
        res = struct.unpack_from(self.form, bytes(buf))
        self.assertEqual(self.md_lv, res[0] >> 5)
        self.assertEqual(self.version, res[0] & 31)
        self.assertEqual(self.opcode, res[1])
        self.assertEqual(self.flags, res[2])
        self.assertEqual(self.first_tlv_offset, res[3])
        self.assertEqual(self.transaction_id, res[4])
        self.assertEqual(self.end_tlv, res[5])

    def test_len(self):
        self.assertEqual(9, len(self.ins))

    def test_default_args(self):
        ins = cfm.loopback_reply()
        buf = ins.serialize()
        res = struct.unpack_from(cfm.loopback_reply._PACK_STR, bytes(buf))
        self.assertEqual(res[0] >> 5, 0)
        self.assertEqual(res[0] & 31, 0)
        self.assertEqual(res[1], 2)
        self.assertEqual(res[2], 0)
        self.assertEqual(res[3], 4)
        self.assertEqual(res[4], 0)