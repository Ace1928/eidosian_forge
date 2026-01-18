import abc
import struct
from os_ken.lib import addrconv
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
@sctp.register_chunk_type
class chunk_sack(chunk):
    """Stream Control Transmission Protocol (SCTP)
    sub encoder/decoder class for Selective Acknowledgement (SACK) chunk
    (RFC 4960).

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
    tsn_ack        TSN of the last DATA chunk received in sequence
                   before a gap.
    a_rwnd         Advertised Receiver Window Credit.
    gapack_num     number of Gap Ack blocks.
    duptsn_num     number of duplicate TSNs.
    gapacks        a list of Gap Ack blocks. one block is made of a list
                   with the start offset and the end offset from tsn_ack.
                   e.g.) gapacks = [[2, 3], [10, 12], [19, 21]]
    duptsns        a list of duplicate TSN.
    ============== =====================================================
    """
    _PACK_STR = '!BBHIIHH'
    _MIN_LEN = struct.calcsize(_PACK_STR)
    _GAPACK_STR = '!HH'
    _GAPACK_LEN = struct.calcsize(_GAPACK_STR)
    _DUPTSN_STR = '!I'
    _DUPTSN_LEN = struct.calcsize(_DUPTSN_STR)

    @classmethod
    def chunk_type(cls):
        return TYPE_SACK

    def __init__(self, flags=0, length=0, tsn_ack=0, a_rwnd=0, gapack_num=0, duptsn_num=0, gapacks=None, duptsns=None):
        super(chunk_sack, self).__init__(self.chunk_type(), length)
        self.flags = flags
        self.tsn_ack = tsn_ack
        self.a_rwnd = a_rwnd
        self.gapack_num = gapack_num
        self.duptsn_num = duptsn_num
        gapacks = gapacks or []
        assert isinstance(gapacks, list)
        for one in gapacks:
            assert isinstance(one, list)
            assert 2 == len(one)
        self.gapacks = gapacks
        duptsns = duptsns or []
        assert isinstance(duptsns, list)
        self.duptsns = duptsns

    @classmethod
    def parser(cls, buf):
        _, flags, length, tsn_ack, a_rwnd, gapack_num, duptsn_num = struct.unpack_from(cls._PACK_STR, buf)
        gapacks = []
        offset = cls._MIN_LEN
        for _ in range(gapack_num):
            gapack_start, gapack_end = struct.unpack_from(cls._GAPACK_STR, buf, offset)
            gapacks.append([gapack_start, gapack_end])
            offset += cls._GAPACK_LEN
        duptsns = []
        for _ in range(duptsn_num):
            duptsn, = struct.unpack_from(cls._DUPTSN_STR, buf, offset)
            duptsns.append(duptsn)
            offset += cls._DUPTSN_LEN
        return cls(flags, length, tsn_ack, a_rwnd, gapack_num, duptsn_num, gapacks, duptsns)

    def serialize(self):
        buf = bytearray(struct.pack(self._PACK_STR, self.chunk_type(), self.flags, self.length, self.tsn_ack, self.a_rwnd, self.gapack_num, self.duptsn_num))
        for one in self.gapacks:
            buf.extend(struct.pack(chunk_sack._GAPACK_STR, one[0], one[1]))
        for one in self.duptsns:
            buf.extend(struct.pack(chunk_sack._DUPTSN_STR, one))
        if 0 == self.length:
            self.length = len(buf)
            struct.pack_into('!H', buf, 2, self.length)
        return bytes(buf)