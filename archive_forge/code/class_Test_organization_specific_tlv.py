import unittest
import logging
import inspect
import struct
from os_ken.lib import addrconv
from os_ken.lib.packet import cfm
class Test_organization_specific_tlv(unittest.TestCase):

    def setUp(self):
        self._type = cfm.CFM_ORGANIZATION_SPECIFIC_TLV
        self.length = 10
        self.oui = b'\xff\x124'
        self.subtype = 3
        self.value = b'\x01\x02\x0f\x0e\r\x0c'
        self.ins = cfm.organization_specific_tlv(self.length, self.oui, self.subtype, self.value)
        self.form = '!BH3sB6s'
        self.buf = struct.pack(self.form, self._type, self.length, self.oui, self.subtype, self.value)

    def tearDown(self):
        pass

    def test_init(self):
        self.assertEqual(self.length, self.ins.length)
        self.assertEqual(self.oui, self.ins.oui)
        self.assertEqual(self.subtype, self.ins.subtype)
        self.assertEqual(self.value, self.ins.value)

    def test_parser(self):
        _res = cfm.organization_specific_tlv.parser(self.buf)
        if type(_res) is tuple:
            res = _res[0]
        else:
            res = _res
        self.assertEqual(self.length, res.length)
        self.assertEqual(self.oui, res.oui)
        self.assertEqual(self.subtype, res.subtype)
        self.assertEqual(self.value, res.value)

    def test_serialize(self):
        buf = self.ins.serialize()
        res = struct.unpack_from(self.form, bytes(buf))
        self.assertEqual(self._type, res[0])
        self.assertEqual(self.length, res[1])
        self.assertEqual(self.oui, res[2])
        self.assertEqual(self.subtype, res[3])
        self.assertEqual(self.value, res[4])

    def test_serialize_with_zero(self):
        ins = cfm.organization_specific_tlv(0, self.oui, self.subtype, self.value)
        buf = ins.serialize()
        res = struct.unpack_from(self.form, bytes(buf))
        self.assertEqual(self._type, res[0])
        self.assertEqual(self.length, res[1])
        self.assertEqual(self.oui, res[2])
        self.assertEqual(self.subtype, res[3])
        self.assertEqual(self.value, res[4])

    def test_len(self):
        self.assertEqual(1 + 2 + 10, len(self.ins))

    def test_default_args(self):
        ins = cfm.organization_specific_tlv()
        buf = ins.serialize()
        res = struct.unpack_from(cfm.organization_specific_tlv._PACK_STR, bytes(buf))
        self.assertEqual(res[0], cfm.CFM_ORGANIZATION_SPECIFIC_TLV)
        self.assertEqual(res[1], 4)
        self.assertEqual(res[2], b'\x00\x00\x00')
        self.assertEqual(res[3], 0)