import abc
import struct
from os_ken.lib import addrconv
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
@sctp.register_chunk_type
class chunk_data(chunk):
    """Stream Control Transmission Protocol (SCTP)
    sub encoder/decoder class for Payload Data (DATA) chunk (RFC 4960).

    This class is used with the following.

    - os_ken.lib.packet.sctp.sctp

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte order.
    __init__ takes the corresponding args in this order.

    .. tabularcolumns:: |l|L|

    ============== =====================================================
    Attribute      Description
    ============== =====================================================
    unordered      if set to '1', the receiver ignores the sequence number.
    begin          if set to '1', this chunk is the first fragment.
    end            if set to '1', this chunk is the last fragment.
    length         length of this chunk containing this header.
                   (0 means automatically-calculate when encoding)
    tsn            Transmission Sequence Number.
    sid            stream id.
    seq            the sequence number.
    payload_id     application specified protocol id. '0' means that
                   no application id is identified.
    payload_data   user data.
    ============== =====================================================
    """
    _PACK_STR = '!BBHIHHI'
    _MIN_LEN = struct.calcsize(_PACK_STR)

    @classmethod
    def chunk_type(cls):
        return TYPE_DATA

    def __init__(self, unordered=0, begin=0, end=0, length=0, tsn=0, sid=0, seq=0, payload_id=0, payload_data=None):
        assert 1 == unordered | 1
        assert 1 == begin | 1
        assert 1 == end | 1
        assert payload_data is not None
        super(chunk_data, self).__init__(self.chunk_type(), length)
        self.unordered = unordered
        self.begin = begin
        self.end = end
        self.tsn = tsn
        self.sid = sid
        self.seq = seq
        self.payload_id = payload_id
        self.payload_data = payload_data

    @classmethod
    def parser(cls, buf):
        _, flags, length, tsn, sid, seq, payload_id = struct.unpack_from(cls._PACK_STR, buf)
        unordered = flags >> 2 & 1
        begin = flags >> 1 & 1
        end = flags >> 0 & 1
        fmt = '!%ds' % (length - cls._MIN_LEN)
        payload_data, = struct.unpack_from(fmt, buf, cls._MIN_LEN)
        return cls(unordered, begin, end, length, tsn, sid, seq, payload_id, payload_data)

    def serialize(self):
        flags = self.unordered << 2 | self.begin << 1 | self.end << 0
        buf = bytearray(struct.pack(self._PACK_STR, self.chunk_type(), flags, self.length, self.tsn, self.sid, self.seq, self.payload_id))
        buf.extend(self.payload_data)
        if 0 == self.length:
            self.length = len(buf)
            struct.pack_into('!H', buf, 2, self.length)
        return bytes(buf)