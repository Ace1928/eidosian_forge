import unittest
import logging
import struct
import inspect
from os_ken.ofproto import inet
from os_ken.lib.packet import ipv4
from os_ken.lib.packet import ipv6
from os_ken.lib.packet import packet
from os_ken.lib.packet import packet_utils
from os_ken.lib.packet import vrrp
from os_ken.lib import addrconv
class Test_vrrpv2(unittest.TestCase):
    """ Test case for vrrp v2
    """
    version = vrrp.VRRP_VERSION_V2
    type_ = vrrp.VRRP_TYPE_ADVERTISEMENT
    vrid = 128
    priority = 100
    count_ip = 1
    auth_type = vrrp.VRRP_AUTH_NO_AUTH
    max_adver_int = 100
    checksum = 0
    ip_address = '192.168.0.1'
    auth_data = (0, 0)
    vrrpv2 = vrrp.vrrpv2.create(type_, vrid, priority, max_adver_int, [ip_address])
    buf = struct.pack(vrrp.vrrpv2._PACK_STR + '4sII', vrrp.vrrp_to_version_type(vrrp.VRRP_VERSION_V2, type_), vrid, priority, count_ip, auth_type, max_adver_int, checksum, addrconv.ipv4.text_to_bin(ip_address), auth_data[0], auth_data[1])

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_init(self):
        self.assertEqual(self.type_, self.vrrpv2.type)
        self.assertEqual(self.vrid, self.vrrpv2.vrid)
        self.assertEqual(self.priority, self.vrrpv2.priority)
        self.assertEqual(self.count_ip, self.vrrpv2.count_ip)
        self.assertEqual(self.auth_type, self.vrrpv2.auth_type)
        self.assertEqual(1, len(self.vrrpv2.ip_addresses))
        self.assertEqual(self.ip_address, self.vrrpv2.ip_addresses[0])
        self.assertEqual(self.auth_data, self.vrrpv2.auth_data)

    def test_parser(self):
        vrrpv2, _cls, _ = self.vrrpv2.parser(self.buf)
        self.assertEqual(self.version, vrrpv2.version)
        self.assertEqual(self.type_, vrrpv2.type)
        self.assertEqual(self.vrid, vrrpv2.vrid)
        self.assertEqual(self.priority, vrrpv2.priority)
        self.assertEqual(self.count_ip, vrrpv2.count_ip)
        self.assertEqual(self.auth_type, vrrpv2.auth_type)
        self.assertEqual(self.max_adver_int, vrrpv2.max_adver_int)
        self.assertEqual(self.checksum, vrrpv2.checksum)
        self.assertEqual(1, len(vrrpv2.ip_addresses))
        self.assertEqual(str, type(vrrpv2.ip_addresses[0]))
        self.assertEqual(self.ip_address, vrrpv2.ip_addresses[0])
        self.assertEqual(self.auth_data, vrrpv2.auth_data)

    def test_serialize(self):
        src_ip = '192.168.0.1'
        dst_ip = vrrp.VRRP_IPV4_DST_ADDRESS
        prev = ipv4.ipv4(4, 5, 0, 0, 0, 0, 0, vrrp.VRRP_IPV4_TTL, inet.IPPROTO_VRRP, 0, src_ip, dst_ip)
        type_ = vrrp.VRRP_TYPE_ADVERTISEMENT
        vrid = 5
        priority = 10
        max_adver_int = 30
        ip_address = '192.168.0.2'
        ip_addresses = [ip_address]
        vrrp_ = vrrp.vrrpv2.create(type_, vrid, priority, max_adver_int, ip_addresses)
        buf = vrrp_.serialize(bytearray(), prev)
        pack_str = vrrp.vrrpv2._PACK_STR + '4sII'
        pack_len = struct.calcsize(pack_str)
        res = struct.unpack(pack_str, bytes(buf))
        self.assertEqual(res[0], vrrp.vrrp_to_version_type(vrrp.VRRP_VERSION_V2, type_))
        self.assertEqual(res[1], vrid)
        self.assertEqual(res[2], priority)
        self.assertEqual(res[3], len(ip_addresses))
        self.assertEqual(res[4], vrrp.VRRP_AUTH_NO_AUTH)
        self.assertEqual(res[5], max_adver_int)
        self.assertEqual(res[7], addrconv.ipv4.text_to_bin(ip_address))
        self.assertEqual(res[8], 0)
        self.assertEqual(res[9], 0)
        self.assertEqual(len(buf), pack_len)
        s = packet_utils.checksum(buf)
        self.assertEqual(0, s)

    def test_malformed_vrrpv2(self):
        m_short_buf = self.buf[1:vrrp.vrrpv2._MIN_LEN]
        self.assertRaises(Exception, vrrp.vrrp.parser, m_short_buf)

    def test_create_packet(self):
        primary_ip = '192.168.0.2'
        p0 = self.vrrpv2.create_packet(primary_ip)
        p0.serialize()
        p1 = packet.Packet(bytes(p0.data))
        p1.serialize()
        self.assertEqual(p0.data, p1.data)

    def _test_is_valid(self, type_=None, vrid=None, priority=None, max_adver_int=None):
        if type_ is None:
            type_ = self.type_
        if vrid is None:
            vrid = self.vrid
        if priority is None:
            priority = self.priority
        if max_adver_int is None:
            max_adver_int = self.max_adver_int
        vrrp_ = vrrp.vrrpv2.create(type_, vrid, priority, max_adver_int, [self.ip_address])
        return vrrp_.is_valid()

    def test_is_valid_ok(self):
        self.assertTrue(self._test_is_valid())

    def test_is_valid_ng_type(self):
        self.assertTrue(not self._test_is_valid(type_=15))

    def test_is_valid_ng_vrid_min(self):
        vrid = vrrp.VRRP_VRID_MIN - 1
        self.assertTrue(not self._test_is_valid(vrid=vrid))

    def test_is_valid_ng_vrid_max(self):
        vrid = vrrp.VRRP_VRID_MAX + 1
        self.assertTrue(not self._test_is_valid(vrid=vrid))

    def test_is_valid_ng_priority_min(self):
        priority = vrrp.VRRP_PRIORITY_MIN - 1
        self.assertTrue(not self._test_is_valid(priority=priority))

    def test_is_valid_ng_priority_max(self):
        priority = vrrp.VRRP_PRIORITY_MAX + 1
        self.assertTrue(not self._test_is_valid(priority=priority))

    def test_is_valid_ng_adver_min(self):
        max_adver_int = vrrp.VRRP_V2_MAX_ADVER_INT_MIN - 1
        self.assertTrue(not self._test_is_valid(max_adver_int=max_adver_int))

    def test_is_valid_ng_adver_max(self):
        max_adver_int = vrrp.VRRP_V2_MAX_ADVER_INT_MAX + 1
        self.assertTrue(not self._test_is_valid(max_adver_int=max_adver_int))

    def test_to_string(self):
        vrrpv2_values = {'version': self.version, 'type': self.type_, 'vrid': self.vrid, 'priority': self.priority, 'count_ip': self.count_ip, 'max_adver_int': self.max_adver_int, 'checksum': self.vrrpv2.checksum, 'ip_addresses': [self.ip_address], 'auth_type': self.auth_type, 'auth_data': self.auth_data, 'identification': self.vrrpv2.identification}
        _vrrpv2_str = ','.join(['%s=%s' % (k, repr(vrrpv2_values[k])) for k, v in inspect.getmembers(self.vrrpv2) if k in vrrpv2_values])
        vrrpv2_str = '%s(%s)' % (vrrp.vrrpv2.__name__, _vrrpv2_str)
        self.assertEqual(str(self.vrrpv2), vrrpv2_str)
        self.assertEqual(repr(self.vrrpv2), vrrpv2_str)