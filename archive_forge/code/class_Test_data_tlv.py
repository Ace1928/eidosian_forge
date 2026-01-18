import unittest
import logging
import inspect
import struct
from os_ken.lib import addrconv
from os_ken.lib.packet import cfm
class Test_data_tlv(unittest.TestCase):

    def setUp(self):
        self._type = cfm.CFM_DATA_TLV
        self.length = 3
        self.data_value = b'\x01\x02\x03'
        self.ins = cfm.data_tlv(self.length, self.data_value)
        self.form = '!BH3s'
        self.buf = struct.pack(self.form, self._type, self.length, self.data_value)

    def tearDown(self):
        pass

    def test_init(self):
        self.assertEqual(self.length, self.ins.length)
        self.assertEqual(self.data_value, self.ins.data_value)

    def test_parser(self):
        _res = cfm.data_tlv.parser(self.buf)
        if type(_res) is tuple:
            res = _res[0]
        else:
            res = _res
        self.assertEqual(self.length, res.length)
        self.assertEqual(self.data_value, res.data_value)

    def test_serialize(self):
        buf = self.ins.serialize()
        res = struct.unpack_from(self.form, bytes(buf))
        self.assertEqual(self._type, res[0])
        self.assertEqual(self.length, res[1])
        self.assertEqual(self.data_value, res[2])

    def test_serialize_with_length_zero(self):
        ins = cfm.data_tlv(0, self.data_value)
        buf = ins.serialize()
        res = struct.unpack_from(self.form, bytes(buf))
        self.assertEqual(self._type, res[0])
        self.assertEqual(self.length, res[1])
        self.assertEqual(self.data_value, res[2])

    def test_len(self):
        self.assertEqual(1 + 2 + 3, len(self.ins))

    def test_default_args(self):
        ins = cfm.data_tlv()
        buf = ins.serialize()
        res = struct.unpack_from(cfm.data_tlv._PACK_STR, bytes(buf))
        self.assertEqual(res[0], cfm.CFM_DATA_TLV)
        self.assertEqual(res[1], 0)