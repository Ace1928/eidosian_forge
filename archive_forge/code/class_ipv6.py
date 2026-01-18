import abc
import struct
from . import packet_base
from . import icmpv6
from . import tcp
from . import udp
from . import sctp
from . import gre
from . import in_proto as inet
from os_ken.lib import addrconv
from os_ken.lib import stringify
class ipv6(packet_base.PacketBase):
    """IPv6 (RFC 2460) header encoder/decoder class.

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte order.
    IPv6 addresses are represented as a string like 'ff02::1'.
    __init__ takes the corresponding args in this order.

    .. tabularcolumns:: |l|p{30em}|l|

    ============== ======================================== ==================
    Attribute      Description                              Example
    ============== ======================================== ==================
    version        Version
    traffic_class  Traffic Class
    flow_label     When decoding, Flow Label.
                   When encoding, the most significant 8
                   bits of Flow Label.
    payload_length Payload Length
    nxt            Next Header
    hop_limit      Hop Limit
    src            Source Address                           'ff02::1'
    dst            Destination Address                      '::'
    ext_hdrs       Extension Headers
    ============== ======================================== ==================
    """
    _PACK_STR = '!IHBB16s16s'
    _MIN_LEN = struct.calcsize(_PACK_STR)
    _IPV6_EXT_HEADER_TYPE = {}
    _TYPE = {'ascii': ['src', 'dst']}

    @staticmethod
    def register_header_type(type_):

        def _register_header_type(cls):
            ipv6._IPV6_EXT_HEADER_TYPE[type_] = cls
            return cls
        return _register_header_type

    def __init__(self, version=6, traffic_class=0, flow_label=0, payload_length=0, nxt=inet.IPPROTO_TCP, hop_limit=255, src='10::10', dst='20::20', ext_hdrs=None):
        super(ipv6, self).__init__()
        self.version = version
        self.traffic_class = traffic_class
        self.flow_label = flow_label
        self.payload_length = payload_length
        self.nxt = nxt
        self.hop_limit = hop_limit
        self.src = src
        self.dst = dst
        ext_hdrs = ext_hdrs or []
        assert isinstance(ext_hdrs, list)
        for ext_hdr in ext_hdrs:
            assert isinstance(ext_hdr, header)
        self.ext_hdrs = ext_hdrs

    @classmethod
    def parser(cls, buf):
        v_tc_flow, payload_length, nxt, hlim, src, dst = struct.unpack_from(cls._PACK_STR, buf)
        version = v_tc_flow >> 28
        traffic_class = v_tc_flow >> 20 & 255
        flow_label = v_tc_flow & 1048575
        hop_limit = hlim
        offset = cls._MIN_LEN
        last = nxt
        ext_hdrs = []
        while True:
            cls_ = cls._IPV6_EXT_HEADER_TYPE.get(last)
            if not cls_:
                break
            hdr = cls_.parser(buf[offset:])
            ext_hdrs.append(hdr)
            offset += len(hdr)
            last = hdr.nxt
        msg = cls(version, traffic_class, flow_label, payload_length, nxt, hop_limit, addrconv.ipv6.bin_to_text(src), addrconv.ipv6.bin_to_text(dst), ext_hdrs)
        return (msg, ipv6.get_packet_type(last), buf[offset:offset + payload_length])

    def serialize(self, payload, prev):
        hdr = bytearray(40)
        v_tc_flow = self.version << 28 | self.traffic_class << 20 | self.flow_label
        struct.pack_into(ipv6._PACK_STR, hdr, 0, v_tc_flow, self.payload_length, self.nxt, self.hop_limit, addrconv.ipv6.text_to_bin(self.src), addrconv.ipv6.text_to_bin(self.dst))
        if self.ext_hdrs:
            for ext_hdr in self.ext_hdrs:
                hdr.extend(ext_hdr.serialize())
        if 0 == self.payload_length:
            payload_length = len(payload)
            for ext_hdr in self.ext_hdrs:
                payload_length += len(ext_hdr)
            self.payload_length = payload_length
            struct.pack_into('!H', hdr, 4, self.payload_length)
        return hdr

    def __len__(self):
        ext_hdrs_len = 0
        for ext_hdr in self.ext_hdrs:
            ext_hdrs_len += len(ext_hdr)
        return self._MIN_LEN + ext_hdrs_len