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
class Test_icmpv6_membership_query(unittest.TestCase):
    type_ = 130
    code = 0
    csum = 46500
    maxresp = 10000
    address = 'ff08::1'
    buf = b"\x82\x00\xb5\xa4'\x10\x00\x00" + b'\xff\x08\x00\x00\x00\x00\x00\x00' + b'\x00\x00\x00\x00\x00\x00\x00\x01'

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_init(self):
        mld = icmpv6.mld(self.maxresp, self.address)
        self.assertEqual(mld.maxresp, self.maxresp)
        self.assertEqual(mld.address, self.address)

    def test_parser(self):
        msg, n, _ = icmpv6.icmpv6.parser(self.buf)
        self.assertEqual(msg.type_, self.type_)
        self.assertEqual(msg.code, self.code)
        self.assertEqual(msg.csum, self.csum)
        self.assertEqual(msg.data.maxresp, self.maxresp)
        self.assertEqual(msg.data.address, self.address)
        self.assertEqual(n, None)

    def test_serialize(self):
        src_ipv6 = '3ffe:507:0:1:200:86ff:fe05:80da'
        dst_ipv6 = '3ffe:501:0:1001::2'
        prev = ipv6(6, 0, 0, len(self.buf), 64, 255, src_ipv6, dst_ipv6)
        mld_csum = icmpv6_csum(prev, self.buf)
        mld = icmpv6.mld(self.maxresp, self.address)
        icmp = icmpv6.icmpv6(self.type_, self.code, 0, mld)
        buf = bytes(icmp.serialize(bytearray(), prev))
        type_, code, csum = struct.unpack_from(icmp._PACK_STR, buf, 0)
        maxresp, address = struct.unpack_from(mld._PACK_STR, buf, icmp._MIN_LEN)
        self.assertEqual(type_, self.type_)
        self.assertEqual(code, self.code)
        self.assertEqual(csum, mld_csum)
        self.assertEqual(maxresp, self.maxresp)
        self.assertEqual(address, addrconv.ipv6.text_to_bin(self.address))

    def test_to_string(self):
        ml = icmpv6.mld(self.maxresp, self.address)
        ic = icmpv6.icmpv6(self.type_, self.code, self.csum, ml)
        mld_values = {'maxresp': self.maxresp, 'address': self.address}
        _mld_str = ','.join(['%s=%s' % (k, repr(mld_values[k])) for k, v in inspect.getmembers(ml) if k in mld_values])
        mld_str = '%s(%s)' % (icmpv6.mld.__name__, _mld_str)
        icmp_values = {'type_': repr(self.type_), 'code': repr(self.code), 'csum': repr(self.csum), 'data': mld_str}
        _ic_str = ','.join(['%s=%s' % (k, icmp_values[k]) for k, v in inspect.getmembers(ic) if k in icmp_values])
        ic_str = '%s(%s)' % (icmpv6.icmpv6.__name__, _ic_str)
        self.assertEqual(str(ic), ic_str)
        self.assertEqual(repr(ic), ic_str)

    def test_default_args(self):
        prev = ipv6(nxt=inet.IPPROTO_ICMPV6)
        ic = icmpv6.icmpv6(type_=icmpv6.MLD_LISTENER_QUERY, data=icmpv6.mld())
        prev.serialize(ic, None)
        buf = ic.serialize(bytearray(), prev)
        res = struct.unpack(icmpv6.icmpv6._PACK_STR, bytes(buf[:4]))
        self.assertEqual(res[0], icmpv6.MLD_LISTENER_QUERY)
        self.assertEqual(res[1], 0)
        self.assertEqual(res[2], icmpv6_csum(prev, buf))
        res = struct.unpack(icmpv6.mld._PACK_STR, bytes(buf[4:]))
        self.assertEqual(res[0], 0)
        self.assertEqual(res[1], addrconv.ipv6.text_to_bin('::'))

    def test_json(self):
        ic1 = icmpv6.icmpv6(type_=icmpv6.MLD_LISTENER_QUERY, data=icmpv6.mld())
        jsondict = ic1.to_jsondict()
        ic2 = icmpv6.icmpv6.from_jsondict(jsondict['icmpv6'])
        self.assertEqual(str(ic1), str(ic2))