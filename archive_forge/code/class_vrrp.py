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
class vrrp(packet_base.PacketBase):
    """The base class for VRRPv2 (RFC 3768) and VRRPv3 (RFC 5798)
    header encoder/decoder classes.

    Unlike other os_ken.lib.packet.packet_base.PacketBase derived classes,
    This class should not be directly instantiated by user.

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte order.

    ============== ====================
    Attribute      Description
    ============== ====================
    version        Version
    type           Type
    vrid           Virtual Rtr ID (VRID)
    priority       Priority
    count_ip       Count IPvX Addr.                    Calculated automatically when encoding.
    max_adver_int  Maximum Advertisement Interval (Max Adver Int)
    checksum       Checksum.                    Calculated automatically when encoding.
    ip_addresses   IPvX Address(es).  A python list of IP addresses.
    auth_type      Authentication Type (only for VRRPv2)
    auth_data      Authentication Data (only for VRRPv2)
    ============== ====================
    """
    _VERSION_PACK_STR = '!B'
    _IPV4_ADDRESS_PACK_STR_RAW = '4s'
    _IPV4_ADDRESS_PACK_STR = '!' + _IPV4_ADDRESS_PACK_STR_RAW
    _IPV4_ADDRESS_LEN = struct.calcsize(_IPV4_ADDRESS_PACK_STR)
    _IPV6_ADDRESS_LEN = 16
    _IPV6_ADDRESS_PACK_STR_RAW = '%ds' % _IPV6_ADDRESS_LEN
    _IPV6_ADDRESS_PACK_STR = '!' + _IPV6_ADDRESS_PACK_STR_RAW
    _IPV6_ADDRESS_LEN = struct.calcsize(_IPV6_ADDRESS_PACK_STR)
    _VRRP_VERSIONS = {}
    _SEC_IN_MAX_ADVER_INT_UNIT = {}

    @staticmethod
    def get_payload(packet_):
        may_ip = None
        may_vrrp = None
        idx = 0
        for protocol in packet_:
            if isinstance(protocol, ipv4.ipv4) or isinstance(protocol, ipv6.ipv6):
                may_ip = protocol
                try:
                    if isinstance(packet_.protocols[idx + 1], vrrp):
                        may_vrrp = packet_.protocols[idx + 1]
                finally:
                    break
            idx += 1
        if may_ip and may_vrrp:
            return (may_ip, may_vrrp)
        else:
            return (None, None)

    @classmethod
    def register_vrrp_version(cls, version, sec_in_max_adver_int_unit):

        def _register_vrrp_version(cls_):
            cls._VRRP_VERSIONS[version] = cls_
            cls._SEC_IN_MAX_ADVER_INT_UNIT[version] = sec_in_max_adver_int_unit
            return cls_
        return _register_vrrp_version

    @staticmethod
    def sec_to_max_adver_int(version, seconds):
        return int(seconds * vrrp._SEC_IN_MAX_ADVER_INT_UNIT[version])

    @staticmethod
    def max_adver_int_to_sec(version, max_adver_int):
        return float(max_adver_int) / vrrp._SEC_IN_MAX_ADVER_INT_UNIT[version]

    def __init__(self, version, type_, vrid, priority, count_ip, max_adver_int, checksum, ip_addresses, auth_type=None, auth_data=None):
        super(vrrp, self).__init__()
        self.version = version
        self.type = type_
        self.vrid = vrid
        self.priority = priority
        self.count_ip = count_ip
        self.max_adver_int = max_adver_int
        self.checksum = checksum
        self.ip_addresses = ip_addresses
        assert len(list(ip_addresses)) == self.count_ip
        self.auth_type = auth_type
        self.auth_data = auth_data
        self._is_ipv6 = is_ipv6(list(self.ip_addresses)[0])
        self.identification = 0

    def checksum_ok(self, ipvx, vrrp_buf):
        cls_ = self._VRRP_VERSIONS[self.version]
        return cls_.checksum_ok(self, ipvx, vrrp_buf)

    @property
    def max_adver_int_in_sec(self):
        return self.max_adver_int_to_sec(self.version, self.max_adver_int)

    @property
    def is_ipv6(self):
        return self._is_ipv6

    def __len__(self):
        cls_ = self._VRRP_VERSIONS[self.version]
        return cls_.__len__(self)

    @staticmethod
    def create_version(version, type_, vrid, priority, max_adver_int, ip_addresses, auth_type=None, auth_data=None):
        cls_ = vrrp._VRRP_VERSIONS.get(version, None)
        if not cls_:
            raise ValueError('unknown VRRP version %d' % version)
        if priority is None:
            priority = VRRP_PRIORITY_BACKUP_DEFAULT
        count_ip = len(ip_addresses)
        if max_adver_int is None:
            max_adver_int = cls_.sec_to_max_adver_int(VRRP_MAX_ADVER_INT_DEFAULT_IN_SEC)
        return cls_(version, type_, vrid, priority, count_ip, max_adver_int, None, ip_addresses, auth_type=auth_type, auth_data=auth_data)

    def get_identification(self):
        self.identification += 1
        self.identification &= 65535
        if self.identification == 0:
            self.identification += 1
            self.identification &= 65535
        return self.identification

    def create_packet(self, primary_ip_address, vlan_id=None):
        """Prepare a VRRP packet.

        Returns a newly created os_ken.lib.packet.packet.Packet object
        with appropriate protocol header objects added by add_protocol().
        It's caller's responsibility to serialize().
        The serialized packet would looks like the ones described in
        the following sections.

        * RFC 3768 5.1. VRRP Packet Format
        * RFC 5798 5.1. VRRP Packet Format

        ================== ====================
        Argument           Description
        ================== ====================
        primary_ip_address Source IP address
        vlan_id            VLAN ID.  None for no VLAN.
        ================== ====================
        """
        if self.is_ipv6:
            traffic_class = 192
            flow_label = 0
            payload_length = ipv6.ipv6._MIN_LEN + len(self)
            e = ethernet.ethernet(VRRP_IPV6_DST_MAC_ADDRESS, vrrp_ipv6_src_mac_address(self.vrid), ether.ETH_TYPE_IPV6)
            ip = ipv6.ipv6(6, traffic_class, flow_label, payload_length, inet.IPPROTO_VRRP, VRRP_IPV6_HOP_LIMIT, primary_ip_address, VRRP_IPV6_DST_ADDRESS)
        else:
            header_length = ipv4.ipv4._MIN_LEN // 4
            total_length = 0
            tos = 192
            identification = self.get_identification()
            e = ethernet.ethernet(VRRP_IPV4_DST_MAC_ADDRESS, vrrp_ipv4_src_mac_address(self.vrid), ether.ETH_TYPE_IP)
            ip = ipv4.ipv4(4, header_length, tos, total_length, identification, 0, 0, VRRP_IPV4_TTL, inet.IPPROTO_VRRP, 0, primary_ip_address, VRRP_IPV4_DST_ADDRESS)
        p = packet.Packet()
        p.add_protocol(e)
        if vlan_id is not None:
            vlan_ = vlan.vlan(0, 0, vlan_id, e.ethertype)
            e.ethertype = ether.ETH_TYPE_8021Q
            p.add_protocol(vlan_)
        p.add_protocol(ip)
        p.add_protocol(self)
        return p

    @classmethod
    def parser(cls, buf):
        version_type, = struct.unpack_from(cls._VERSION_PACK_STR, buf)
        version, _type = vrrp_from_version_type(version_type)
        cls_ = cls._VRRP_VERSIONS[version]
        return cls_.parser(buf)

    @staticmethod
    def serialize_static(vrrp_, prev):
        assert isinstance(vrrp_, vrrp)
        cls = vrrp._VRRP_VERSIONS[vrrp_.version]
        return cls.serialize_static(vrrp_, prev)

    def serialize(self, payload, prev):
        return self.serialize_static(self, prev)

    @staticmethod
    def is_valid_ttl(ipvx):
        version = ipvx.version
        if version == 4:
            return ipvx.ttl == VRRP_IPV4_TTL
        if version == 6:
            return ipvx.hop_limit == VRRP_IPV6_HOP_LIMIT
        raise ValueError('invalid ip version %d' % version)

    def is_valid(self):
        cls = self._VRRP_VERSIONS.get(self.version, None)
        if cls is None:
            return False
        return cls.is_valid(self)