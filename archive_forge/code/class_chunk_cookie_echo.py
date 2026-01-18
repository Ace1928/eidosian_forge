import abc
import struct
from os_ken.lib import addrconv
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
@sctp.register_chunk_type
class chunk_cookie_echo(chunk):
    """Stream Control Transmission Protocol (SCTP)
    sub encoder/decoder class for Cookie Echo (COOKIE ECHO) chunk (RFC 4960).

    This class is used with the following.

    - os_ken.lib.packet.sctp.sctp

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte order.
    __init__ takes the corresponding args in this order.

    .. tabularcolumns:: |l|L|

    ============== =====================================================
    Attribute      Description
    ============== =====================================================
    flags          set to '0'. this field will be ignored.
    length         length of this chunk containing this header.
                   (0 means automatically-calculate when encoding)
    cookie         cookie data.
    ============== =====================================================
    """
    _PACK_STR = '!BBH'
    _MIN_LEN = struct.calcsize(_PACK_STR)

    @classmethod
    def chunk_type(cls):
        return TYPE_COOKIE_ECHO

    def __init__(self, flags=0, length=0, cookie=None):
        super(chunk_cookie_echo, self).__init__(self.chunk_type(), length)
        self.flags = flags
        self.cookie = cookie

    @classmethod
    def parser(cls, buf):
        _, flags, length = struct.unpack_from(cls._PACK_STR, buf)
        _len = length - cls._MIN_LEN
        cookie = None
        if _len:
            fmt = '%ds' % _len
            cookie, = struct.unpack_from(fmt, buf, cls._MIN_LEN)
        return cls(flags, length, cookie)

    def serialize(self):
        buf = bytearray(struct.pack(self._PACK_STR, self.chunk_type(), self.flags, self.length))
        if self.cookie is not None:
            buf.extend(self.cookie)
        if 0 == self.length:
            self.length = len(buf)
            struct.pack_into('!H', buf, 2, self.length)
        mod = len(buf) % 4
        if mod:
            buf.extend(bytearray(4 - mod))
        return bytes(buf)