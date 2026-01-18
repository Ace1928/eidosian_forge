import unittest
import inspect
import logging
from struct import pack, unpack_from, pack_into
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.lib.packet.ethernet import ethernet
from os_ken.lib.packet.ipv4 import ipv4
from os_ken.lib.packet.packet import Packet
from os_ken.lib.packet.packet_utils import checksum
from os_ken.lib import addrconv
from os_ken.lib.packet.igmp import igmp
from os_ken.lib.packet.igmp import igmpv3_query
from os_ken.lib.packet.igmp import igmpv3_report
from os_ken.lib.packet.igmp import igmpv3_report_group
from os_ken.lib.packet.igmp import IGMP_TYPE_QUERY
from os_ken.lib.packet.igmp import IGMP_TYPE_REPORT_V3
from os_ken.lib.packet.igmp import MODE_IS_INCLUDE
class Test_igmp(unittest.TestCase):
    """ Test case for Internet Group Management Protocol
    """

    def setUp(self):
        self.msgtype = IGMP_TYPE_QUERY
        self.maxresp = 100
        self.csum = 0
        self.address = '225.0.0.1'
        self.buf = pack(igmp._PACK_STR, self.msgtype, self.maxresp, self.csum, addrconv.ipv4.text_to_bin(self.address))
        self.g = igmp(self.msgtype, self.maxresp, self.csum, self.address)

    def tearDown(self):
        pass

    def find_protocol(self, pkt, name):
        for p in pkt.protocols:
            if p.protocol_name == name:
                return p

    def test_init(self):
        self.assertEqual(self.msgtype, self.g.msgtype)
        self.assertEqual(self.maxresp, self.g.maxresp)
        self.assertEqual(self.csum, self.g.csum)
        self.assertEqual(self.address, self.g.address)

    def test_parser(self):
        _res = self.g.parser(self.buf)
        if type(_res) is tuple:
            res = _res[0]
        else:
            res = _res
        self.assertEqual(res.msgtype, self.msgtype)
        self.assertEqual(res.maxresp, self.maxresp)
        self.assertEqual(res.csum, self.csum)
        self.assertEqual(res.address, self.address)

    def test_serialize(self):
        data = bytearray()
        prev = None
        buf = self.g.serialize(data, prev)
        res = unpack_from(igmp._PACK_STR, bytes(buf))
        self.assertEqual(res[0], self.msgtype)
        self.assertEqual(res[1], self.maxresp)
        self.assertEqual(res[2], checksum(self.buf))
        self.assertEqual(res[3], addrconv.ipv4.text_to_bin(self.address))

    def _build_igmp(self):
        dl_dst = '11:22:33:44:55:66'
        dl_src = 'aa:bb:cc:dd:ee:ff'
        dl_type = ether.ETH_TYPE_IP
        e = ethernet(dl_dst, dl_src, dl_type)
        total_length = 20 + igmp._MIN_LEN
        nw_proto = inet.IPPROTO_IGMP
        nw_dst = '11.22.33.44'
        nw_src = '55.66.77.88'
        i = ipv4(total_length=total_length, src=nw_src, dst=nw_dst, proto=nw_proto)
        p = Packet()
        p.add_protocol(e)
        p.add_protocol(i)
        p.add_protocol(self.g)
        p.serialize()
        return p

    def test_build_igmp(self):
        p = self._build_igmp()
        e = self.find_protocol(p, 'ethernet')
        self.assertIsNotNone(e)
        self.assertEqual(e.ethertype, ether.ETH_TYPE_IP)
        i = self.find_protocol(p, 'ipv4')
        self.assertTrue(i)
        self.assertEqual(i.proto, inet.IPPROTO_IGMP)
        g = self.find_protocol(p, 'igmp')
        self.assertTrue(g)
        self.assertEqual(g.msgtype, self.msgtype)
        self.assertEqual(g.maxresp, self.maxresp)
        self.assertEqual(g.csum, checksum(self.buf))
        self.assertEqual(g.address, self.address)

    def test_to_string(self):
        igmp_values = {'msgtype': repr(self.msgtype), 'maxresp': repr(self.maxresp), 'csum': repr(self.csum), 'address': repr(self.address)}
        _g_str = ','.join(['%s=%s' % (k, igmp_values[k]) for k, v in inspect.getmembers(self.g) if k in igmp_values])
        g_str = '%s(%s)' % (igmp.__name__, _g_str)
        self.assertEqual(str(self.g), g_str)
        self.assertEqual(repr(self.g), g_str)

    def test_malformed_igmp(self):
        m_short_buf = self.buf[1:igmp._MIN_LEN]
        self.assertRaises(Exception, igmp.parser, m_short_buf)

    def test_default_args(self):
        ig = igmp()
        buf = ig.serialize(bytearray(), None)
        res = unpack_from(igmp._PACK_STR, bytes(buf))
        self.assertEqual(res[0], 17)
        self.assertEqual(res[1], 0)
        self.assertEqual(res[3], addrconv.ipv4.text_to_bin('0.0.0.0'))

    def test_json(self):
        jsondict = self.g.to_jsondict()
        g = igmp.from_jsondict(jsondict['igmp'])
        self.assertEqual(str(self.g), str(g))