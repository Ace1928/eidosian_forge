import unittest
import logging
import inspect
import struct
from os_ken.lib import addrconv
from os_ken.lib.packet import cfm
class Test_cc_message(unittest.TestCase):

    def setUp(self):
        self.md_lv = 1
        self.version = 1
        self.opcode = cfm.CFM_CC_MESSAGE
        self.rdi = 1
        self.interval = 5
        self.first_tlv_offset = cfm.cc_message._TLV_OFFSET
        self.seq_num = 2
        self.mep_id = 2
        self.md_name_format = cfm.cc_message._MD_FMT_CHARACTER_STRING
        self.md_name_length = 3
        self.md_name = b'foo'
        self.short_ma_name_format = 2
        self.short_ma_name_length = 8
        self.short_ma_name = b'hogehoge'
        self.tlvs = []
        self.end_tlv = 0
        self.ins = cfm.cc_message(self.md_lv, self.version, self.rdi, self.interval, self.seq_num, self.mep_id, self.md_name_format, self.md_name_length, self.md_name, self.short_ma_name_format, self.short_ma_name_length, self.short_ma_name, self.tlvs)
        self.form = '!4BIH2B3s2B8s33x12x4xB'
        self.buf = struct.pack(self.form, self.md_lv << 5 | self.version, self.opcode, self.rdi << 7 | self.interval, self.first_tlv_offset, self.seq_num, self.mep_id, self.md_name_format, self.md_name_length, self.md_name, self.short_ma_name_format, self.short_ma_name_length, self.short_ma_name, self.end_tlv)

    def tearDown(self):
        pass

    def test_init(self):
        self.assertEqual(self.md_lv, self.ins.md_lv)
        self.assertEqual(self.version, self.ins.version)
        self.assertEqual(self.rdi, self.ins.rdi)
        self.assertEqual(self.interval, self.ins.interval)
        self.assertEqual(self.seq_num, self.ins.seq_num)
        self.assertEqual(self.mep_id, self.ins.mep_id)
        self.assertEqual(self.md_name_format, self.ins.md_name_format)
        self.assertEqual(self.md_name_length, self.ins.md_name_length)
        self.assertEqual(self.md_name, self.ins.md_name)
        self.assertEqual(self.short_ma_name_format, self.ins.short_ma_name_format)
        self.assertEqual(self.short_ma_name_length, self.ins.short_ma_name_length)
        self.assertEqual(self.short_ma_name, self.ins.short_ma_name)
        self.assertEqual(self.tlvs, self.ins.tlvs)

    def test_parser(self):
        _res = cfm.cc_message.parser(self.buf)
        if type(_res) is tuple:
            res = _res[0]
        else:
            res = _res
        self.assertEqual(self.md_lv, res.md_lv)
        self.assertEqual(self.version, res.version)
        self.assertEqual(self.rdi, res.rdi)
        self.assertEqual(self.interval, res.interval)
        self.assertEqual(self.seq_num, res.seq_num)
        self.assertEqual(self.mep_id, res.mep_id)
        self.assertEqual(self.md_name_format, res.md_name_format)
        self.assertEqual(self.md_name_length, res.md_name_length)
        self.assertEqual(self.md_name, res.md_name)
        self.assertEqual(self.short_ma_name_format, res.short_ma_name_format)
        self.assertEqual(self.short_ma_name_length, res.short_ma_name_length)
        self.assertEqual(self.short_ma_name, res.short_ma_name)
        self.assertEqual(self.tlvs, res.tlvs)

    def test_parser_with_no_maintenance_domain_name_present(self):
        form = '!4BIH3B8s37x12x4xB'
        buf = struct.pack(form, self.md_lv << 5 | self.version, self.opcode, self.rdi << 7 | self.interval, self.first_tlv_offset, self.seq_num, self.mep_id, cfm.cc_message._MD_FMT_NO_MD_NAME_PRESENT, self.short_ma_name_format, self.short_ma_name_length, self.short_ma_name, self.end_tlv)
        _res = cfm.cc_message.parser(buf)
        if type(_res) is tuple:
            res = _res[0]
        else:
            res = _res
        self.assertEqual(self.md_lv, res.md_lv)
        self.assertEqual(self.version, res.version)
        self.assertEqual(self.rdi, res.rdi)
        self.assertEqual(self.interval, res.interval)
        self.assertEqual(self.seq_num, res.seq_num)
        self.assertEqual(self.mep_id, res.mep_id)
        self.assertEqual(cfm.cc_message._MD_FMT_NO_MD_NAME_PRESENT, res.md_name_format)
        self.assertEqual(self.short_ma_name_format, res.short_ma_name_format)
        self.assertEqual(self.short_ma_name_length, res.short_ma_name_length)
        self.assertEqual(self.short_ma_name, res.short_ma_name)
        self.assertEqual(self.tlvs, res.tlvs)

    def test_serialize(self):
        buf = self.ins.serialize()
        res = struct.unpack_from(self.form, bytes(buf))
        self.assertEqual(self.md_lv, res[0] >> 5)
        self.assertEqual(self.version, res[0] & 31)
        self.assertEqual(self.opcode, res[1])
        self.assertEqual(self.rdi, res[2] >> 7)
        self.assertEqual(self.interval, res[2] & 7)
        self.assertEqual(self.first_tlv_offset, res[3])
        self.assertEqual(self.seq_num, res[4])
        self.assertEqual(self.mep_id, res[5])
        self.assertEqual(self.md_name_format, res[6])
        self.assertEqual(self.md_name_length, res[7])
        self.assertEqual(self.md_name, res[8])
        self.assertEqual(self.short_ma_name_format, res[9])
        self.assertEqual(self.short_ma_name_length, res[10])
        self.assertEqual(self.short_ma_name, res[11])
        self.assertEqual(self.end_tlv, res[12])

    def test_serialize_with_md_name_length_zero(self):
        ins = cfm.cc_message(self.md_lv, self.version, self.rdi, self.interval, self.seq_num, self.mep_id, self.md_name_format, 0, self.md_name, self.short_ma_name_format, 0, self.short_ma_name, self.tlvs)
        buf = ins.serialize()
        res = struct.unpack_from(self.form, bytes(buf))
        self.assertEqual(self.md_lv, res[0] >> 5)
        self.assertEqual(self.version, res[0] & 31)
        self.assertEqual(self.opcode, res[1])
        self.assertEqual(self.rdi, res[2] >> 7)
        self.assertEqual(self.interval, res[2] & 7)
        self.assertEqual(self.first_tlv_offset, res[3])
        self.assertEqual(self.seq_num, res[4])
        self.assertEqual(self.mep_id, res[5])
        self.assertEqual(self.md_name_format, res[6])
        self.assertEqual(self.md_name_length, res[7])
        self.assertEqual(self.md_name, res[8])
        self.assertEqual(self.short_ma_name_format, res[9])
        self.assertEqual(self.short_ma_name_length, res[10])
        self.assertEqual(self.short_ma_name, res[11])
        self.assertEqual(self.end_tlv, res[12])

    def test_serialize_with_no_maintenance_domain_name_present(self):
        form = '!4BIH3B8s37x12x4xB'
        ins = cfm.cc_message(self.md_lv, self.version, self.rdi, self.interval, self.seq_num, self.mep_id, cfm.cc_message._MD_FMT_NO_MD_NAME_PRESENT, 0, self.md_name, self.short_ma_name_format, 0, self.short_ma_name, self.tlvs)
        buf = ins.serialize()
        res = struct.unpack_from(form, bytes(buf))
        self.assertEqual(self.md_lv, res[0] >> 5)
        self.assertEqual(self.version, res[0] & 31)
        self.assertEqual(self.opcode, res[1])
        self.assertEqual(self.rdi, res[2] >> 7)
        self.assertEqual(self.interval, res[2] & 7)
        self.assertEqual(self.first_tlv_offset, res[3])
        self.assertEqual(self.seq_num, res[4])
        self.assertEqual(self.mep_id, res[5])
        self.assertEqual(cfm.cc_message._MD_FMT_NO_MD_NAME_PRESENT, res[6])
        self.assertEqual(self.short_ma_name_format, res[7])
        self.assertEqual(self.short_ma_name_length, res[8])
        self.assertEqual(self.short_ma_name, res[9])
        self.assertEqual(self.end_tlv, res[10])

    def test_len(self):
        self.assertEqual(75, len(self.ins))

    def test_default_args(self):
        ins = cfm.cc_message()
        buf = ins.serialize()
        res = struct.unpack_from(cfm.cc_message._PACK_STR, bytes(buf))
        self.assertEqual(res[0] >> 5, 0)
        self.assertEqual(res[0] & 31, 0)
        self.assertEqual(res[1], 1)
        self.assertEqual(res[2] >> 7, 0)
        self.assertEqual(res[2] & 7, 4)
        self.assertEqual(res[3], 70)
        self.assertEqual(res[4], 0)
        self.assertEqual(res[5], 1)
        self.assertEqual(res[6], 4)