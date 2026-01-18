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
class Test_mldv2_report(unittest.TestCase):
    type_ = 143
    code = 0
    csum = 46500
    record_num = 0
    records = []
    mld = icmpv6.mldv2_report(record_num, records)
    buf = b'\x8f\x00\xb5\xa4\x00\x00\x00\x00'

    def setUp(self):
        pass

    def setUp_with_records(self):
        self.record1 = icmpv6.mldv2_report_group(icmpv6.MODE_IS_INCLUDE, 0, 0, 'ff00::1')
        self.record2 = icmpv6.mldv2_report_group(icmpv6.MODE_IS_INCLUDE, 0, 2, 'ff00::2', ['fe80::1', 'fe80::2'])
        self.record3 = icmpv6.mldv2_report_group(icmpv6.MODE_IS_INCLUDE, 1, 0, 'ff00::3', [], b'abc\x00')
        self.record4 = icmpv6.mldv2_report_group(icmpv6.MODE_IS_INCLUDE, 2, 2, 'ff00::4', ['fe80::1', 'fe80::2'], b'abcde\x00\x00\x00')
        self.records = [self.record1, self.record2, self.record3, self.record4]
        self.record_num = len(self.records)
        self.mld = icmpv6.mldv2_report(self.record_num, self.records)
        self.buf = b'\x8f\x00\xb5\xa4\x00\x00\x00\x04' + b'\x01\x00\x00\x00' + b'\xff\x00\x00\x00\x00\x00\x00\x00' + b'\x00\x00\x00\x00\x00\x00\x00\x01' + b'\x01\x00\x00\x02' + b'\xff\x00\x00\x00\x00\x00\x00\x00' + b'\x00\x00\x00\x00\x00\x00\x00\x02' + b'\xfe\x80\x00\x00\x00\x00\x00\x00' + b'\x00\x00\x00\x00\x00\x00\x00\x01' + b'\xfe\x80\x00\x00\x00\x00\x00\x00' + b'\x00\x00\x00\x00\x00\x00\x00\x02' + b'\x01\x01\x00\x00' + b'\xff\x00\x00\x00\x00\x00\x00\x00' + b'\x00\x00\x00\x00\x00\x00\x00\x03' + b'abc\x00' + b'\x01\x02\x00\x02' + b'\xff\x00\x00\x00\x00\x00\x00\x00' + b'\x00\x00\x00\x00\x00\x00\x00\x04' + b'\xfe\x80\x00\x00\x00\x00\x00\x00' + b'\x00\x00\x00\x00\x00\x00\x00\x01' + b'\xfe\x80\x00\x00\x00\x00\x00\x00' + b'\x00\x00\x00\x00\x00\x00\x00\x02' + b'abcde\x00\x00\x00'

    def tearDown(self):
        pass

    def find_protocol(self, pkt, name):
        for p in pkt.protocols:
            if p.protocol_name == name:
                return p

    def test_init(self):
        self.assertEqual(self.mld.record_num, self.record_num)
        self.assertEqual(self.mld.records, self.records)

    def test_init_with_records(self):
        self.setUp_with_records()
        self.test_init()

    def test_parser(self):
        msg, n, _ = icmpv6.icmpv6.parser(self.buf)
        self.assertEqual(msg.type_, self.type_)
        self.assertEqual(msg.code, self.code)
        self.assertEqual(msg.csum, self.csum)
        self.assertEqual(msg.data.record_num, self.record_num)
        self.assertEqual(repr(msg.data.records), repr(self.records))

    def test_parser_with_records(self):
        self.setUp_with_records()
        self.test_parser()

    def test_serialize(self):
        src_ipv6 = '3ffe:507:0:1:200:86ff:fe05:80da'
        dst_ipv6 = '3ffe:501:0:1001::2'
        prev = ipv6(6, 0, 0, len(self.buf), 64, 255, src_ipv6, dst_ipv6)
        mld_csum = icmpv6_csum(prev, self.buf)
        icmp = icmpv6.icmpv6(self.type_, self.code, 0, self.mld)
        buf = icmp.serialize(bytearray(), prev)
        type_, code, csum = struct.unpack_from(icmp._PACK_STR, bytes(buf))
        record_num, = struct.unpack_from(self.mld._PACK_STR, bytes(buf), icmp._MIN_LEN)
        self.assertEqual(type_, self.type_)
        self.assertEqual(code, self.code)
        self.assertEqual(csum, mld_csum)
        self.assertEqual(record_num, self.record_num)

    def test_serialize_with_records(self):
        self.setUp_with_records()
        src_ipv6 = '3ffe:507:0:1:200:86ff:fe05:80da'
        dst_ipv6 = '3ffe:501:0:1001::2'
        prev = ipv6(6, 0, 0, len(self.buf), 64, 255, src_ipv6, dst_ipv6)
        mld_csum = icmpv6_csum(prev, self.buf)
        icmp = icmpv6.icmpv6(self.type_, self.code, 0, self.mld)
        buf = bytes(icmp.serialize(bytearray(), prev))
        type_, code, csum = struct.unpack_from(icmp._PACK_STR, bytes(buf))
        record_num, = struct.unpack_from(self.mld._PACK_STR, bytes(buf), icmp._MIN_LEN)
        offset = icmp._MIN_LEN + self.mld._MIN_LEN
        rec1 = icmpv6.mldv2_report_group.parser(buf[offset:])
        offset += len(rec1)
        rec2 = icmpv6.mldv2_report_group.parser(buf[offset:])
        offset += len(rec2)
        rec3 = icmpv6.mldv2_report_group.parser(buf[offset:])
        offset += len(rec3)
        rec4 = icmpv6.mldv2_report_group.parser(buf[offset:])
        self.assertEqual(type_, self.type_)
        self.assertEqual(code, self.code)
        self.assertEqual(csum, mld_csum)
        self.assertEqual(record_num, self.record_num)
        self.assertEqual(repr(rec1), repr(self.record1))
        self.assertEqual(repr(rec2), repr(self.record2))
        self.assertEqual(repr(rec3), repr(self.record3))
        self.assertEqual(repr(rec4), repr(self.record4))

    def _build_mldv2_report(self):
        e = ethernet(ethertype=ether.ETH_TYPE_IPV6)
        i = ipv6(nxt=inet.IPPROTO_ICMPV6)
        ic = icmpv6.icmpv6(type_=icmpv6.MLDV2_LISTENER_REPORT, data=self.mld)
        p = e / i / ic
        return p

    def test_build_mldv2_report(self):
        p = self._build_mldv2_report()
        e = self.find_protocol(p, 'ethernet')
        self.assertTrue(e)
        self.assertEqual(e.ethertype, ether.ETH_TYPE_IPV6)
        i = self.find_protocol(p, 'ipv6')
        self.assertTrue(i)
        self.assertEqual(i.nxt, inet.IPPROTO_ICMPV6)
        ic = self.find_protocol(p, 'icmpv6')
        self.assertTrue(ic)
        self.assertEqual(ic.type_, icmpv6.MLDV2_LISTENER_REPORT)
        self.assertEqual(ic.data.record_num, self.record_num)
        self.assertEqual(ic.data.records, self.records)

    def test_build_mldv2_report_with_records(self):
        self.setUp_with_records()
        self.test_build_mldv2_report()

    def test_to_string(self):
        ic = icmpv6.icmpv6(self.type_, self.code, self.csum, self.mld)
        mld_values = {'record_num': self.record_num, 'records': self.records}
        _mld_str = ','.join(['%s=%s' % (k, repr(mld_values[k])) for k, v in inspect.getmembers(self.mld) if k in mld_values])
        mld_str = '%s(%s)' % (icmpv6.mldv2_report.__name__, _mld_str)
        icmp_values = {'type_': repr(self.type_), 'code': repr(self.code), 'csum': repr(self.csum), 'data': mld_str}
        _ic_str = ','.join(['%s=%s' % (k, icmp_values[k]) for k, v in inspect.getmembers(ic) if k in icmp_values])
        ic_str = '%s(%s)' % (icmpv6.icmpv6.__name__, _ic_str)
        self.assertEqual(str(ic), ic_str)
        self.assertEqual(repr(ic), ic_str)

    def test_to_string_with_records(self):
        self.setUp_with_records()
        self.test_to_string()

    def test_record_num_larger_than_records(self):
        self.record1 = icmpv6.mldv2_report_group(icmpv6.MODE_IS_INCLUDE, 0, 0, 'ff00::1')
        self.record2 = icmpv6.mldv2_report_group(icmpv6.MODE_IS_INCLUDE, 0, 2, 'ff00::2', ['fe80::1', 'fe80::2'])
        self.record3 = icmpv6.mldv2_report_group(icmpv6.MODE_IS_INCLUDE, 1, 0, 'ff00::3', [], b'abc\x00')
        self.record4 = icmpv6.mldv2_report_group(icmpv6.MODE_IS_INCLUDE, 2, 2, 'ff00::4', ['fe80::1', 'fe80::2'], b'abcde\x00\x00\x00')
        self.records = [self.record1, self.record2, self.record3, self.record4]
        self.record_num = len(self.records) + 1
        self.buf = struct.pack(icmpv6.mldv2_report._PACK_STR, self.record_num)
        self.buf += self.record1.serialize()
        self.buf += self.record2.serialize()
        self.buf += self.record3.serialize()
        self.buf += self.record4.serialize()
        self.mld = icmpv6.mldv2_report(self.record_num, self.records)
        self.assertRaises(AssertionError, self.test_parser)

    def test_record_num_smaller_than_records(self):
        self.record1 = icmpv6.mldv2_report_group(icmpv6.MODE_IS_INCLUDE, 0, 0, 'ff00::1')
        self.record2 = icmpv6.mldv2_report_group(icmpv6.MODE_IS_INCLUDE, 0, 2, 'ff00::2', ['fe80::1', 'fe80::2'])
        self.record3 = icmpv6.mldv2_report_group(icmpv6.MODE_IS_INCLUDE, 1, 0, 'ff00::3', [], b'abc\x00')
        self.record4 = icmpv6.mldv2_report_group(icmpv6.MODE_IS_INCLUDE, 2, 2, 'ff00::4', ['fe80::1', 'fe80::2'], b'abcde\x00\x00\x00')
        self.records = [self.record1, self.record2, self.record3, self.record4]
        self.record_num = len(self.records) - 1
        self.buf = struct.pack(icmpv6.mldv2_report._PACK_STR, self.record_num)
        self.buf += self.record1.serialize()
        self.buf += self.record2.serialize()
        self.buf += self.record3.serialize()
        self.buf += self.record4.serialize()
        self.mld = icmpv6.mldv2_report(self.record_num, self.records)
        self.assertRaises(AssertionError, self.test_parser)

    def test_default_args(self):
        prev = ipv6(nxt=inet.IPPROTO_ICMPV6)
        ic = icmpv6.icmpv6(type_=icmpv6.MLDV2_LISTENER_REPORT, data=icmpv6.mldv2_report())
        prev.serialize(ic, None)
        buf = ic.serialize(bytearray(), prev)
        res = struct.unpack(icmpv6.icmpv6._PACK_STR, bytes(buf[:4]))
        self.assertEqual(res[0], icmpv6.MLDV2_LISTENER_REPORT)
        self.assertEqual(res[1], 0)
        self.assertEqual(res[2], icmpv6_csum(prev, buf))
        res = struct.unpack(icmpv6.mldv2_report._PACK_STR, bytes(buf[4:]))
        self.assertEqual(res[0], 0)
        record1 = icmpv6.mldv2_report_group(icmpv6.MODE_IS_INCLUDE, 0, 0, 'ff00::1')
        record2 = icmpv6.mldv2_report_group(icmpv6.MODE_IS_INCLUDE, 0, 2, 'ff00::2', ['fe80::1', 'fe80::2'])
        records = [record1, record2]
        rep = icmpv6.mldv2_report(records=records)
        buf = rep.serialize()
        res = struct.unpack_from(icmpv6.mldv2_report._PACK_STR, bytes(buf))
        self.assertEqual(res[0], len(records))
        res = struct.unpack_from(icmpv6.mldv2_report_group._PACK_STR, bytes(buf), icmpv6.mldv2_report._MIN_LEN)
        self.assertEqual(res[0], icmpv6.MODE_IS_INCLUDE)
        self.assertEqual(res[1], 0)
        self.assertEqual(res[2], 0)
        self.assertEqual(res[3], addrconv.ipv6.text_to_bin('ff00::1'))
        res = struct.unpack_from(icmpv6.mldv2_report_group._PACK_STR, bytes(buf), icmpv6.mldv2_report._MIN_LEN + icmpv6.mldv2_report_group._MIN_LEN)
        self.assertEqual(res[0], icmpv6.MODE_IS_INCLUDE)
        self.assertEqual(res[1], 0)
        self.assertEqual(res[2], 2)
        self.assertEqual(res[3], addrconv.ipv6.text_to_bin('ff00::2'))
        res = struct.unpack_from('16s16s', bytes(buf), icmpv6.mldv2_report._MIN_LEN + icmpv6.mldv2_report_group._MIN_LEN + icmpv6.mldv2_report_group._MIN_LEN)
        self.assertEqual(res[0], addrconv.ipv6.text_to_bin('fe80::1'))
        self.assertEqual(res[1], addrconv.ipv6.text_to_bin('fe80::2'))

    def test_json(self):
        jsondict = self.mld.to_jsondict()
        mld = icmpv6.mldv2_report.from_jsondict(jsondict['mldv2_report'])
        self.assertEqual(str(self.mld), str(mld))

    def test_json_with_records(self):
        self.setUp_with_records()
        self.test_json()