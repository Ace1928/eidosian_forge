import struct
from math import trunc
from os_ken.lib import addrconv
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
from os_ken.lib.packet import packet_utils
class igmpv3_report_group(stringify.StringifyMixin):
    """
    Internet Group Management Protocol(IGMP, RFC 3376)
    Membership Report Group Record message encoder/decoder class.

    http://www.ietf.org/rfc/rfc3376.txt

    This is used with os_ken.lib.packet.igmp.igmpv3_report.

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte
    order.
    __init__ takes the corresponding args in this order.

    .. tabularcolumns:: |l|L|

    =============== ====================================================
    Attribute       Description
    =============== ====================================================
    type\\_          a group record type for v3.
    aux_len         the length of the auxiliary data.
    num             a number of the multicast servers.
    address         a group address value.
    srcs            a list of IPv4 addresses of the multicast servers.
    aux             the auxiliary data.
    =============== ====================================================
    """
    _PACK_STR = '!BBH4s'
    _MIN_LEN = struct.calcsize(_PACK_STR)
    _TYPE = {'ascii': ['address'], 'asciilist': ['srcs']}

    def __init__(self, type_=0, aux_len=0, num=0, address='0.0.0.0', srcs=None, aux=None):
        self.type_ = type_
        self.aux_len = aux_len
        self.num = num
        self.address = address
        srcs = srcs or []
        assert isinstance(srcs, list)
        for src in srcs:
            assert isinstance(src, str)
        self.srcs = srcs
        self.aux = aux

    @classmethod
    def parser(cls, buf):
        type_, aux_len, num, address = struct.unpack_from(cls._PACK_STR, buf)
        offset = cls._MIN_LEN
        srcs = []
        while 0 < len(buf[offset:]) and num > len(srcs):
            assert 4 <= len(buf[offset:])
            src, = struct.unpack_from('4s', buf, offset)
            srcs.append(addrconv.ipv4.bin_to_text(src))
            offset += 4
        assert num == len(srcs)
        aux = None
        if aux_len:
            aux, = struct.unpack_from('%ds' % (aux_len * 4), buf, offset)
        return cls(type_, aux_len, num, addrconv.ipv4.bin_to_text(address), srcs, aux)

    def serialize(self):
        buf = bytearray(struct.pack(self._PACK_STR, self.type_, self.aux_len, self.num, addrconv.ipv4.text_to_bin(self.address)))
        for src in self.srcs:
            buf.extend(struct.pack('4s', addrconv.ipv4.text_to_bin(src)))
        if 0 == self.num:
            self.num = len(self.srcs)
            struct.pack_into('!H', buf, 2, self.num)
        if self.aux is not None:
            mod = len(self.aux) % 4
            if mod:
                self.aux += bytearray(4 - mod)
                self.aux = bytes(self.aux)
            buf.extend(self.aux)
            if 0 == self.aux_len:
                self.aux_len = len(self.aux) // 4
                struct.pack_into('!B', buf, 1, self.aux_len)
        return bytes(buf)

    def __len__(self):
        return self._MIN_LEN + len(self.srcs) * 4 + self.aux_len * 4