import unittest
import logging
import struct
import inspect
from os_ken.ofproto import ether, inet
from os_ken.lib.packet.ethernet import ethernet
from os_ken.lib.packet import icmpv6
from os_ken.lib.packet.ipv6 import ipv6
from os_ken.lib.packet import packet_utils
from os_ken.lib import addrconv
class Test_mldv2_report_group(unittest.TestCase):
    type_ = icmpv6.MODE_IS_INCLUDE
    aux_len = 0
    num = 0
    address = 'ff00::1'
    srcs = []
    aux = None
    mld = icmpv6.mldv2_report_group(type_, aux_len, num, address, srcs, aux)
    buf = b'\x01\x00\x00\x00' + b'\xff\x00\x00\x00\x00\x00\x00\x00' + b'\x00\x00\x00\x00\x00\x00\x00\x01'

    def setUp(self):
        pass

    def setUp_with_srcs(self):
        self.srcs = ['fe80::1', 'fe80::2', 'fe80::3']
        self.num = len(self.srcs)
        self.mld = icmpv6.mldv2_report_group(self.type_, self.aux_len, self.num, self.address, self.srcs, self.aux)
        self.buf = b'\x01\x00\x00\x03' + b'\xff\x00\x00\x00\x00\x00\x00\x00' + b'\x00\x00\x00\x00\x00\x00\x00\x01' + b'\xfe\x80\x00\x00\x00\x00\x00\x00' + b'\x00\x00\x00\x00\x00\x00\x00\x01' + b'\xfe\x80\x00\x00\x00\x00\x00\x00' + b'\x00\x00\x00\x00\x00\x00\x00\x02' + b'\xfe\x80\x00\x00\x00\x00\x00\x00' + b'\x00\x00\x00\x00\x00\x00\x00\x03'

    def setUp_with_aux(self):
        self.aux = b'\x01\x02\x03\x04\x05\x06\x07\x08'
        self.aux_len = len(self.aux) // 4
        self.mld = icmpv6.mldv2_report_group(self.type_, self.aux_len, self.num, self.address, self.srcs, self.aux)
        self.buf = b'\x01\x02\x00\x00' + b'\xff\x00\x00\x00\x00\x00\x00\x00' + b'\x00\x00\x00\x00\x00\x00\x00\x01' + b'\x01\x02\x03\x04\x05\x06\x07\x08'

    def setUp_with_srcs_and_aux(self):
        self.srcs = ['fe80::1', 'fe80::2', 'fe80::3']
        self.num = len(self.srcs)
        self.aux = b'\x01\x02\x03\x04\x05\x06\x07\x08'
        self.aux_len = len(self.aux) // 4
        self.mld = icmpv6.mldv2_report_group(self.type_, self.aux_len, self.num, self.address, self.srcs, self.aux)
        self.buf = b'\x01\x02\x00\x03' + b'\xff\x00\x00\x00\x00\x00\x00\x00' + b'\x00\x00\x00\x00\x00\x00\x00\x01' + b'\xfe\x80\x00\x00\x00\x00\x00\x00' + b'\x00\x00\x00\x00\x00\x00\x00\x01' + b'\xfe\x80\x00\x00\x00\x00\x00\x00' + b'\x00\x00\x00\x00\x00\x00\x00\x02' + b'\xfe\x80\x00\x00\x00\x00\x00\x00' + b'\x00\x00\x00\x00\x00\x00\x00\x03' + b'\x01\x02\x03\x04\x05\x06\x07\x08'

    def tearDown(self):
        pass

    def test_init(self):
        self.assertEqual(self.mld.type_, self.type_)
        self.assertEqual(self.mld.aux_len, self.aux_len)
        self.assertEqual(self.mld.num, self.num)
        self.assertEqual(self.mld.address, self.address)
        self.assertEqual(self.mld.srcs, self.srcs)
        self.assertEqual(self.mld.aux, self.aux)

    def test_init_with_srcs(self):
        self.setUp_with_srcs()
        self.test_init()

    def test_init_with_aux(self):
        self.setUp_with_aux()
        self.test_init()

    def test_init_with_srcs_and_aux(self):
        self.setUp_with_srcs_and_aux()
        self.test_init()

    def test_parser(self):
        _res = icmpv6.mldv2_report_group.parser(self.buf)
        if type(_res) is tuple:
            res = _res[0]
        else:
            res = _res
        self.assertEqual(res.type_, self.type_)
        self.assertEqual(res.aux_len, self.aux_len)
        self.assertEqual(res.num, self.num)
        self.assertEqual(res.address, self.address)
        self.assertEqual(res.srcs, self.srcs)
        self.assertEqual(res.aux, self.aux)

    def test_parser_with_srcs(self):
        self.setUp_with_srcs()
        self.test_parser()

    def test_parser_with_aux(self):
        self.setUp_with_aux()
        self.test_parser()

    def test_parser_with_srcs_and_aux(self):
        self.setUp_with_srcs_and_aux()
        self.test_parser()

    def test_serialize(self):
        buf = self.mld.serialize()
        res = struct.unpack_from(icmpv6.mldv2_report_group._PACK_STR, bytes(buf))
        self.assertEqual(res[0], self.type_)
        self.assertEqual(res[1], self.aux_len)
        self.assertEqual(res[2], self.num)
        self.assertEqual(res[3], addrconv.ipv6.text_to_bin(self.address))

    def test_serialize_with_srcs(self):
        self.setUp_with_srcs()
        buf = self.mld.serialize()
        res = struct.unpack_from(icmpv6.mldv2_report_group._PACK_STR, bytes(buf))
        src1, src2, src3 = struct.unpack_from('16s16s16s', bytes(buf), icmpv6.mldv2_report_group._MIN_LEN)
        self.assertEqual(res[0], self.type_)
        self.assertEqual(res[1], self.aux_len)
        self.assertEqual(res[2], self.num)
        self.assertEqual(res[3], addrconv.ipv6.text_to_bin(self.address))
        self.assertEqual(src1, addrconv.ipv6.text_to_bin(self.srcs[0]))
        self.assertEqual(src2, addrconv.ipv6.text_to_bin(self.srcs[1]))
        self.assertEqual(src3, addrconv.ipv6.text_to_bin(self.srcs[2]))

    def test_serialize_with_aux(self):
        self.setUp_with_aux()
        buf = self.mld.serialize()
        res = struct.unpack_from(icmpv6.mldv2_report_group._PACK_STR, bytes(buf))
        aux, = struct.unpack_from('%ds' % (self.aux_len * 4), bytes(buf), icmpv6.mldv2_report_group._MIN_LEN)
        self.assertEqual(res[0], self.type_)
        self.assertEqual(res[1], self.aux_len)
        self.assertEqual(res[2], self.num)
        self.assertEqual(res[3], addrconv.ipv6.text_to_bin(self.address))
        self.assertEqual(aux, self.aux)

    def test_serialize_with_srcs_and_aux(self):
        self.setUp_with_srcs_and_aux()
        buf = self.mld.serialize()
        res = struct.unpack_from(icmpv6.mldv2_report_group._PACK_STR, bytes(buf))
        src1, src2, src3 = struct.unpack_from('16s16s16s', bytes(buf), icmpv6.mldv2_report_group._MIN_LEN)
        aux, = struct.unpack_from('%ds' % (self.aux_len * 4), bytes(buf), icmpv6.mldv2_report_group._MIN_LEN + 16 * 3)
        self.assertEqual(res[0], self.type_)
        self.assertEqual(res[1], self.aux_len)
        self.assertEqual(res[2], self.num)
        self.assertEqual(res[3], addrconv.ipv6.text_to_bin(self.address))
        self.assertEqual(src1, addrconv.ipv6.text_to_bin(self.srcs[0]))
        self.assertEqual(src2, addrconv.ipv6.text_to_bin(self.srcs[1]))
        self.assertEqual(src3, addrconv.ipv6.text_to_bin(self.srcs[2]))
        self.assertEqual(aux, self.aux)

    def test_to_string(self):
        igmp_values = {'type_': repr(self.type_), 'aux_len': repr(self.aux_len), 'num': repr(self.num), 'address': repr(self.address), 'srcs': repr(self.srcs), 'aux': repr(self.aux)}
        _g_str = ','.join(['%s=%s' % (k, igmp_values[k]) for k, v in inspect.getmembers(self.mld) if k in igmp_values])
        g_str = '%s(%s)' % (icmpv6.mldv2_report_group.__name__, _g_str)
        self.assertEqual(str(self.mld), g_str)
        self.assertEqual(repr(self.mld), g_str)

    def test_to_string_with_srcs(self):
        self.setUp_with_srcs()
        self.test_to_string()

    def test_to_string_with_aux(self):
        self.setUp_with_aux()
        self.test_to_string()

    def test_to_string_with_srcs_and_aux(self):
        self.setUp_with_srcs_and_aux()
        self.test_to_string()

    def test_len(self):
        self.assertEqual(len(self.mld), 20)

    def test_len_with_srcs(self):
        self.setUp_with_srcs()
        self.assertEqual(len(self.mld), 68)

    def test_len_with_aux(self):
        self.setUp_with_aux()
        self.assertEqual(len(self.mld), 28)

    def test_len_with_srcs_and_aux(self):
        self.setUp_with_srcs_and_aux()
        self.assertEqual(len(self.mld), 76)

    def test_num_larger_than_srcs(self):
        self.srcs = ['fe80::1', 'fe80::2', 'fe80::3']
        self.num = len(self.srcs) + 1
        self.buf = struct.pack(icmpv6.mldv2_report_group._PACK_STR, self.type_, self.aux_len, self.num, addrconv.ipv6.text_to_bin(self.address))
        for src in self.srcs:
            self.buf += struct.pack('16s', addrconv.ipv6.text_to_bin(src))
        self.mld = icmpv6.mldv2_report_group(self.type_, self.aux_len, self.num, self.address, self.srcs, self.aux)
        self.assertRaises(AssertionError, self.test_parser)

    def test_num_smaller_than_srcs(self):
        self.srcs = ['fe80::1', 'fe80::2', 'fe80::3']
        self.num = len(self.srcs) - 1
        self.buf = struct.pack(icmpv6.mldv2_report_group._PACK_STR, self.type_, self.aux_len, self.num, addrconv.ipv6.text_to_bin(self.address))
        for src in self.srcs:
            self.buf += struct.pack('16s', addrconv.ipv6.text_to_bin(src))
        self.mld = icmpv6.mldv2_report_group(self.type_, self.aux_len, self.num, self.address, self.srcs, self.aux)
        self.assertRaises(AssertionError, self.test_parser)

    def test_aux_len_larger_than_aux(self):
        self.aux = b'\x01\x02\x03\x04\x05\x06\x07\x08'
        self.aux_len = len(self.aux) // 4 + 1
        self.buf = struct.pack(icmpv6.mldv2_report_group._PACK_STR, self.type_, self.aux_len, self.num, addrconv.ipv6.text_to_bin(self.address))
        self.buf += self.aux
        self.mld = icmpv6.mldv2_report_group(self.type_, self.aux_len, self.num, self.address, self.srcs, self.aux)
        self.assertRaises(struct.error, self.test_parser)

    def test_aux_len_smaller_than_aux(self):
        self.aux = b'\x01\x02\x03\x04\x05\x06\x07\x08'
        self.aux_len = len(self.aux) // 4 - 1
        self.buf = struct.pack(icmpv6.mldv2_report_group._PACK_STR, self.type_, self.aux_len, self.num, addrconv.ipv6.text_to_bin(self.address))
        self.buf += self.aux
        self.mld = icmpv6.mldv2_report_group(self.type_, self.aux_len, self.num, self.address, self.srcs, self.aux)
        self.assertRaises(AssertionError, self.test_parser)

    def test_default_args(self):
        rep = icmpv6.mldv2_report_group()
        buf = rep.serialize()
        res = struct.unpack_from(icmpv6.mldv2_report_group._PACK_STR, bytes(buf))
        self.assertEqual(res[0], 0)
        self.assertEqual(res[1], 0)
        self.assertEqual(res[2], 0)
        self.assertEqual(res[3], addrconv.ipv6.text_to_bin('::'))
        srcs = ['fe80::1', 'fe80::2', 'fe80::3']
        rep = icmpv6.mldv2_report_group(srcs=srcs)
        buf = rep.serialize()
        LOG.info(repr(buf))
        res = struct.unpack_from(icmpv6.mldv2_report_group._PACK_STR, bytes(buf))
        self.assertEqual(res[0], 0)
        self.assertEqual(res[1], 0)
        self.assertEqual(res[2], len(srcs))
        self.assertEqual(res[3], addrconv.ipv6.text_to_bin('::'))
        src1, src2, src3 = struct.unpack_from('16s16s16s', bytes(buf), icmpv6.mldv2_report_group._MIN_LEN)
        self.assertEqual(src1, addrconv.ipv6.text_to_bin(srcs[0]))
        self.assertEqual(src2, addrconv.ipv6.text_to_bin(srcs[1]))
        self.assertEqual(src3, addrconv.ipv6.text_to_bin(srcs[2]))
        rep = icmpv6.mldv2_report_group(aux=b'\x01\x02\x03')
        buf = rep.serialize()
        res = struct.unpack_from(icmpv6.mldv2_report_group._PACK_STR, bytes(buf))
        self.assertEqual(res[0], 0)
        self.assertEqual(res[1], 1)
        self.assertEqual(res[2], 0)
        self.assertEqual(res[3], addrconv.ipv6.text_to_bin('::'))
        self.assertEqual(buf[icmpv6.mldv2_report_group._MIN_LEN:], b'\x01\x02\x03\x00')

    def test_json(self):
        jsondict = self.mld.to_jsondict()
        mld = icmpv6.mldv2_report_group.from_jsondict(jsondict['mldv2_report_group'])
        self.assertEqual(str(self.mld), str(mld))

    def test_json_with_srcs(self):
        self.setUp_with_srcs()
        self.test_json()

    def test_json_with_aux(self):
        self.setUp_with_aux()
        self.test_json()

    def test_json_with_srcs_and_aux(self):
        self.setUp_with_srcs_and_aux()
        self.test_json()