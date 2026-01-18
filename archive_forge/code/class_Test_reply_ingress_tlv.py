import unittest
import logging
import inspect
import struct
from os_ken.lib import addrconv
from os_ken.lib.packet import cfm
class Test_reply_ingress_tlv(unittest.TestCase):

    def setUp(self):
        self._type = cfm.CFM_REPLY_INGRESS_TLV
        self.length = 12
        self.action = 2
        self.mac_address = 'aa:bb:cc:56:34:12'
        self.port_id_length = 3
        self.port_id_subtype = 2
        self.port_id = b'\x01\x04\t'
        self.ins = cfm.reply_ingress_tlv(self.length, self.action, self.mac_address, self.port_id_length, self.port_id_subtype, self.port_id)
        self.form = '!BHB6sBB3s'
        self.buf = struct.pack(self.form, self._type, self.length, self.action, addrconv.mac.text_to_bin(self.mac_address), self.port_id_length, self.port_id_subtype, self.port_id)

    def tearDown(self):
        pass

    def test_init(self):
        self.assertEqual(self.length, self.ins.length)
        self.assertEqual(self.action, self.ins.action)
        self.assertEqual(self.mac_address, self.ins.mac_address)
        self.assertEqual(self.port_id_length, self.ins.port_id_length)
        self.assertEqual(self.port_id_subtype, self.ins.port_id_subtype)
        self.assertEqual(self.port_id, self.ins.port_id)

    def test_parser(self):
        _res = cfm.reply_ingress_tlv.parser(self.buf)
        if type(_res) is tuple:
            res = _res[0]
        else:
            res = _res
        self.assertEqual(self.length, res.length)
        self.assertEqual(self.action, res.action)
        self.assertEqual(self.mac_address, res.mac_address)
        self.assertEqual(self.port_id_length, res.port_id_length)
        self.assertEqual(self.port_id_subtype, res.port_id_subtype)
        self.assertEqual(self.port_id, res.port_id)

    def test_serialize(self):
        buf = self.ins.serialize()
        res = struct.unpack_from(self.form, bytes(buf))
        self.assertEqual(self._type, res[0])
        self.assertEqual(self.length, res[1])
        self.assertEqual(self.action, res[2])
        self.assertEqual(addrconv.mac.text_to_bin(self.mac_address), res[3])
        self.assertEqual(self.port_id_length, res[4])
        self.assertEqual(self.port_id_subtype, res[5])
        self.assertEqual(self.port_id, res[6])

    def test_serialize_with_zero(self):
        ins = cfm.reply_ingress_tlv(0, self.action, self.mac_address, 0, self.port_id_subtype, self.port_id)
        buf = ins.serialize()
        res = struct.unpack_from(self.form, bytes(buf))
        self.assertEqual(self._type, res[0])
        self.assertEqual(self.length, res[1])
        self.assertEqual(self.action, res[2])
        self.assertEqual(addrconv.mac.text_to_bin(self.mac_address), res[3])
        self.assertEqual(self.port_id_length, res[4])
        self.assertEqual(self.port_id_subtype, res[5])
        self.assertEqual(self.port_id, res[6])

    def test_len(self):
        self.assertEqual(1 + 2 + 12, len(self.ins))

    def test_default_args(self):
        ins = cfm.reply_ingress_tlv()
        buf = ins.serialize()
        res = struct.unpack_from(cfm.reply_ingress_tlv._PACK_STR, bytes(buf))
        self.assertEqual(res[0], cfm.CFM_REPLY_INGRESS_TLV)
        self.assertEqual(res[1], 7)
        self.assertEqual(res[2], 1)
        self.assertEqual(res[3], addrconv.mac.text_to_bin('00:00:00:00:00:00'))