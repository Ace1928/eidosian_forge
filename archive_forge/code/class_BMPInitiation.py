import struct
from os_ken.lib import addrconv
from os_ken.lib.packet import packet_base
from os_ken.lib.packet import stream_parser
from os_ken.lib.packet.bgp import BGPMessage
from os_ken.lib.type_desc import TypeDisp
@BMPMessage.register_type(BMP_MSG_INITIATION)
class BMPInitiation(BMPMessage):
    """BMP Initiation Message

    ========================== ===============================================
    Attribute                  Description
    ========================== ===============================================
    version                    Version. this packet lib defines BMP ver. 3
    len                        Length field.  Ignored when encoding.
    type                       Type field.  one of BMP\\_MSG\\_ constants.
    info                       One or more piece of information encoded as a
                               TLV
    ========================== ===============================================
    """
    _TLV_PACK_STR = '!HH'
    _MIN_LEN = struct.calcsize(_TLV_PACK_STR)

    def __init__(self, info, type_=BMP_MSG_INITIATION, len_=None, version=VERSION):
        super(BMPInitiation, self).__init__(type_, len_, version)
        self.info = info

    @classmethod
    def parser(cls, buf):
        info = []
        while len(buf):
            if len(buf) < cls._MIN_LEN:
                raise stream_parser.StreamParser.TooSmallException('%d < %d' % (len(buf), cls._MIN_LEN))
            type_, len_ = struct.unpack_from(cls._TLV_PACK_STR, bytes(buf))
            if len(buf) < cls._MIN_LEN + len_:
                raise stream_parser.StreamParser.TooSmallException('%d < %d' % (len(buf), cls._MIN_LEN + len_))
            value = buf[cls._MIN_LEN:cls._MIN_LEN + len_]
            if type_ == BMP_INIT_TYPE_STRING:
                value = value.decode('utf-8')
            buf = buf[cls._MIN_LEN + len_:]
            info.append({'type': type_, 'len': len_, 'value': value})
        return {'info': info}

    def serialize_tail(self):
        msg = bytearray()
        for v in self.info:
            if v['type'] == BMP_INIT_TYPE_STRING:
                value = v['value'].encode('utf-8')
            else:
                value = v['value']
            v['len'] = len(value)
            msg += struct.pack(self._TLV_PACK_STR, v['type'], v['len'])
            msg += value
        return msg