import struct
from os_ken.lib import addrconv
from os_ken.lib.packet import packet_base
from os_ken.lib.packet import stream_parser
from os_ken.lib.packet.bgp import BGPMessage
from os_ken.lib.type_desc import TypeDisp
class BMPMessage(packet_base.PacketBase, TypeDisp):
    """Base class for BGP Monitoring Protocol messages.

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte
    order.
    __init__ takes the corresponding args in this order.

    ========================== ===============================================
    Attribute                  Description
    ========================== ===============================================
    version                    Version. this packet lib defines BMP ver. 3
    len                        Length field.  Ignored when encoding.
    type                       Type field.  one of BMP\\_MSG\\_ constants.
    ========================== ===============================================
    """
    _HDR_PACK_STR = '!BIB'
    _HDR_LEN = struct.calcsize(_HDR_PACK_STR)

    def __init__(self, type_, len_=None, version=VERSION):
        self.version = version
        self.len = len_
        self.type = type_

    @classmethod
    def parse_header(cls, buf):
        if len(buf) < cls._HDR_LEN:
            raise stream_parser.StreamParser.TooSmallException('%d < %d' % (len(buf), cls._HDR_LEN))
        version, len_, type_ = struct.unpack_from(cls._HDR_PACK_STR, bytes(buf))
        return (version, len_, type_)

    @classmethod
    def parser(cls, buf):
        version, msglen, type_ = cls.parse_header(buf)
        if version != VERSION:
            raise ValueError('not supportted bmp version: %d' % version)
        if len(buf) < msglen:
            raise stream_parser.StreamParser.TooSmallException('%d < %d' % (len(buf), msglen))
        binmsg = buf[cls._HDR_LEN:msglen]
        rest = buf[msglen:]
        subcls = cls._lookup_type(type_)
        if subcls == cls._UNKNOWN_TYPE:
            raise ValueError('unknown bmp type: %d' % type_)
        kwargs = subcls.parser(binmsg)
        return (subcls(len_=msglen, type_=type_, version=version, **kwargs), rest)

    def serialize(self):
        tail = self.serialize_tail()
        self.len = self._HDR_LEN + len(tail)
        hdr = bytearray(struct.pack(self._HDR_PACK_STR, self.version, self.len, self.type))
        return hdr + tail

    def __len__(self):
        buf = self.serialize()
        return len(buf)