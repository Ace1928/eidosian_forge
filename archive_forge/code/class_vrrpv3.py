import struct
from os_ken.lib.packet import ethernet
from os_ken.lib.packet import ether_types as ether
from os_ken.lib.packet import in_proto as inet
from os_ken.lib.packet import ipv4
from os_ken.lib.packet import ipv6
from os_ken.lib.packet import packet
from os_ken.lib.packet import packet_base
from os_ken.lib.packet import packet_utils
from os_ken.lib.packet import vlan
from os_ken.lib import addrconv
@vrrp.register_vrrp_version(VRRP_VERSION_V3, 100)
class vrrpv3(vrrp):
    """VRRPv3 (RFC 5798) header encoder/decoder class.

    Unlike other os_ken.lib.packet.packet_base.PacketBase derived classes,
    *create* method should be used to instantiate an object of this class.
    """
    _PACK_STR = '!BBBBHH'
    _MIN_LEN = struct.calcsize(_PACK_STR)
    _CHECKSUM_PACK_STR = '!H'
    _CHECKSUM_OFFSET = 6

    def __len__(self):
        if self.is_ipv6:
            address_len = self._IPV6_ADDRESS_LEN
        else:
            address_len = self._IPV4_ADDRESS_LEN
        return self._MIN_LEN + address_len * self.count_ip

    def checksum_ok(self, ipvx, vrrp_buf):
        return packet_utils.checksum_ip(ipvx, len(self), vrrp_buf) == 0

    @staticmethod
    def create(type_, vrid, priority, max_adver_int, ip_addresses):
        """Unlike other os_ken.lib.packet.packet_base.PacketBase derived classes,
        this method should be used to instantiate an object of this class.

        This method's arguments are same as os_ken.lib.packet.vrrp.vrrp object's
        attributes of the same name.  (except that *type_* corresponds to
        *type* attribute.)
        """
        return vrrp.create_version(VRRP_VERSION_V3, type_, vrid, priority, max_adver_int, ip_addresses)

    @classmethod
    def parser(cls, buf):
        version_type, vrid, priority, count_ip, max_adver_int, checksum = struct.unpack_from(cls._PACK_STR, buf)
        version, type_ = vrrp_from_version_type(version_type)
        max_adver_int &= VRRP_V3_MAX_ADVER_INT_MASK
        offset = cls._MIN_LEN
        address_len = (len(buf) - offset) // count_ip
        if address_len == cls._IPV4_ADDRESS_LEN:
            pack_str = '!' + cls._IPV4_ADDRESS_PACK_STR_RAW * count_ip
            conv = addrconv.ipv4.bin_to_text
        elif address_len == cls._IPV6_ADDRESS_LEN:
            pack_str = '!' + cls._IPV6_ADDRESS_PACK_STR_RAW * count_ip
            conv = addrconv.ipv6.bin_to_text
        else:
            raise ValueError('unknown address version address_len %d count_ip %d' % (address_len, count_ip))
        ip_addresses_bin = struct.unpack_from(pack_str, buf, offset)
        ip_addresses = [conv(x) for x in ip_addresses_bin]
        msg = cls(version, type_, vrid, priority, count_ip, max_adver_int, checksum, ip_addresses)
        return (msg, None, buf[len(msg):])

    @staticmethod
    def serialize_static(vrrp_, prev):
        if isinstance(prev, ipv4.ipv4):
            assert type(vrrp_.ip_addresses[0]) == str
            conv = addrconv.ipv4.text_to_bin
            ip_address_pack_raw = vrrpv3._IPV4_ADDRESS_PACK_STR_RAW
        elif isinstance(prev, ipv6.ipv6):
            assert type(vrrp_.ip_addresses[0]) == str
            conv = addrconv.ipv6.text_to_bin
            ip_address_pack_raw = vrrpv3._IPV6_ADDRESS_PACK_STR_RAW
        else:
            raise ValueError('Unkown network layer %s' % type(prev))
        ip_addresses_pack_str = '!' + ip_address_pack_raw * vrrp_.count_ip
        ip_addresses_len = struct.calcsize(ip_addresses_pack_str)
        vrrp_len = vrrpv3._MIN_LEN + ip_addresses_len
        checksum = False
        if vrrp_.checksum is None:
            checksum = True
            vrrp_.checksum = 0
        buf = bytearray(vrrp_len)
        assert vrrp_.max_adver_int <= VRRP_V3_MAX_ADVER_INT_MASK
        struct.pack_into(vrrpv3._PACK_STR, buf, 0, vrrp_to_version_type(vrrp_.version, vrrp_.type), vrrp_.vrid, vrrp_.priority, vrrp_.count_ip, vrrp_.max_adver_int, vrrp_.checksum)
        struct.pack_into(ip_addresses_pack_str, buf, vrrpv3._MIN_LEN, *[conv(x) for x in vrrp_.ip_addresses])
        if checksum:
            vrrp_.checksum = packet_utils.checksum_ip(prev, len(buf), buf)
            struct.pack_into(vrrpv3._CHECKSUM_PACK_STR, buf, vrrpv3._CHECKSUM_OFFSET, vrrp_.checksum)
        return buf

    def is_valid(self):
        return self.version == VRRP_VERSION_V3 and self.type == VRRP_TYPE_ADVERTISEMENT and (VRRP_VRID_MIN <= self.vrid) and (self.vrid <= VRRP_VRID_MAX) and (VRRP_PRIORITY_MIN <= self.priority) and (self.priority <= VRRP_PRIORITY_MAX) and (VRRP_V3_MAX_ADVER_INT_MIN <= self.max_adver_int) and (self.max_adver_int <= VRRP_V3_MAX_ADVER_INT_MAX) and (self.count_ip == len(self.ip_addresses))