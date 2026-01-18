import unittest
import logging
import inspect
import struct
from os_ken.lib import addrconv
from os_ken.lib.packet import cfm
class Test_sender_id_tlv(unittest.TestCase):

    def setUp(self):
        self._type = cfm.CFM_SENDER_ID_TLV
        self.length = 10
        self.chassis_id_length = 1
        self.chassis_id_subtype = 3
        self.chassis_id = b'\n'
        self.ma_domain_length = 2
        self.ma_domain = b'\x04\x05'
        self.ma_length = 3
        self.ma = b'\x01\x02\x03'
        self.ins = cfm.sender_id_tlv(self.length, self.chassis_id_length, self.chassis_id_subtype, self.chassis_id, self.ma_domain_length, self.ma_domain, self.ma_length, self.ma)
        self.form = '!BHBB1sB2sB3s'
        self.buf = struct.pack(self.form, self._type, self.length, self.chassis_id_length, self.chassis_id_subtype, self.chassis_id, self.ma_domain_length, self.ma_domain, self.ma_length, self.ma)

    def tearDown(self):
        pass

    def test_init(self):
        self.assertEqual(self.length, self.ins.length)
        self.assertEqual(self.chassis_id_length, self.ins.chassis_id_length)
        self.assertEqual(self.chassis_id_subtype, self.ins.chassis_id_subtype)
        self.assertEqual(self.chassis_id, self.ins.chassis_id)
        self.assertEqual(self.ma_domain_length, self.ins.ma_domain_length)
        self.assertEqual(self.ma_domain, self.ins.ma_domain)
        self.assertEqual(self.ma_length, self.ins.ma_length)
        self.assertEqual(self.ma, self.ins.ma)

    def test_parser(self):
        _res = cfm.sender_id_tlv.parser(self.buf)
        if type(_res) is tuple:
            res = _res[0]
        else:
            res = _res
        self.assertEqual(self.length, res.length)
        self.assertEqual(self.chassis_id_length, res.chassis_id_length)
        self.assertEqual(self.chassis_id_subtype, res.chassis_id_subtype)
        self.assertEqual(self.chassis_id, res.chassis_id)
        self.assertEqual(self.ma_domain_length, res.ma_domain_length)
        self.assertEqual(self.ma_domain, res.ma_domain)
        self.assertEqual(self.ma_length, res.ma_length)
        self.assertEqual(self.ma, res.ma)

    def test_serialize(self):
        buf = self.ins.serialize()
        res = struct.unpack_from(self.form, bytes(buf))
        self.assertEqual(self._type, res[0])
        self.assertEqual(self.length, res[1])
        self.assertEqual(self.chassis_id_length, res[2])
        self.assertEqual(self.chassis_id_subtype, res[3])
        self.assertEqual(self.chassis_id, res[4])
        self.assertEqual(self.ma_domain_length, res[5])
        self.assertEqual(self.ma_domain, res[6])
        self.assertEqual(self.ma_length, res[7])
        self.assertEqual(self.ma, res[8])

    def test_serialize_semi_normal_ptn1(self):
        ins = cfm.sender_id_tlv(chassis_id_subtype=self.chassis_id_subtype, chassis_id=self.chassis_id, ma_domain=self.ma_domain)
        buf = ins.serialize()
        form = '!BHBB1sB2sB'
        res = struct.unpack_from(form, bytes(buf))
        self.assertEqual(self._type, res[0])
        self.assertEqual(7, res[1])
        self.assertEqual(self.chassis_id_length, res[2])
        self.assertEqual(self.chassis_id_subtype, res[3])
        self.assertEqual(self.chassis_id, res[4])
        self.assertEqual(self.ma_domain_length, res[5])
        self.assertEqual(self.ma_domain, res[6])
        self.assertEqual(0, res[7])

    def test_serialize_semi_normal_ptn2(self):
        ins = cfm.sender_id_tlv(ma_domain=self.ma_domain, ma=self.ma)
        buf = ins.serialize()
        form = '!BHBB2sB3s'
        res = struct.unpack_from(form, bytes(buf))
        self.assertEqual(self._type, res[0])
        self.assertEqual(8, res[1])
        self.assertEqual(0, res[2])
        self.assertEqual(self.ma_domain_length, res[3])
        self.assertEqual(self.ma_domain, res[4])
        self.assertEqual(self.ma_length, res[5])
        self.assertEqual(self.ma, res[6])

    def test_serialize_semi_normal_ptn3(self):
        ins = cfm.sender_id_tlv(chassis_id_subtype=self.chassis_id_subtype, chassis_id=self.chassis_id)
        buf = ins.serialize()
        form = '!BHBB1sB'
        res = struct.unpack_from(form, bytes(buf))
        self.assertEqual(self._type, res[0])
        self.assertEqual(4, res[1])
        self.assertEqual(self.chassis_id_length, res[2])
        self.assertEqual(self.chassis_id_subtype, res[3])
        self.assertEqual(self.chassis_id, res[4])
        self.assertEqual(0, res[5])

    def test_serialize_semi_normal_ptn4(self):
        ins = cfm.sender_id_tlv(ma_domain=self.ma_domain)
        buf = ins.serialize()
        form = '!BHBB2sB'
        res = struct.unpack_from(form, bytes(buf))
        self.assertEqual(self._type, res[0])
        self.assertEqual(5, res[1])
        self.assertEqual(0, res[2])
        self.assertEqual(self.ma_domain_length, res[3])
        self.assertEqual(self.ma_domain, res[4])
        self.assertEqual(0, res[5])

    def test_serialize_with_length_zero(self):
        ins = cfm.sender_id_tlv(0, 0, self.chassis_id_subtype, self.chassis_id, 0, self.ma_domain, 0, self.ma)
        buf = ins.serialize()
        res = struct.unpack_from(self.form, bytes(buf))
        self.assertEqual(self._type, res[0])
        self.assertEqual(self.length, res[1])
        self.assertEqual(self.chassis_id_length, res[2])
        self.assertEqual(self.chassis_id_subtype, res[3])
        self.assertEqual(self.chassis_id, res[4])
        self.assertEqual(self.ma_domain_length, res[5])
        self.assertEqual(self.ma_domain, res[6])
        self.assertEqual(self.ma_length, res[7])
        self.assertEqual(self.ma, res[8])

    def test_len(self):
        self.assertEqual(1 + 2 + 10, len(self.ins))

    def test_default_args(self):
        ins = cfm.sender_id_tlv()
        buf = ins.serialize()
        res = struct.unpack_from(cfm.sender_id_tlv._PACK_STR, bytes(buf))
        self.assertEqual(res[0], cfm.CFM_SENDER_ID_TLV)
        self.assertEqual(res[1], 1)
        self.assertEqual(res[2], 0)