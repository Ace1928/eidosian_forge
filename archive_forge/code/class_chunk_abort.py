import abc
import struct
from os_ken.lib import addrconv
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
@sctp.register_chunk_type
class chunk_abort(chunk):
    """Stream Control Transmission Protocol (SCTP)
    sub encoder/decoder class for Abort Association (ABORT) chunk (RFC 4960).

    This class is used with the following.

    - os_ken.lib.packet.sctp.sctp

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte order.
    __init__ takes the corresponding args in this order.

    .. tabularcolumns:: |l|L|

    ============== =====================================================
    Attribute      Description
    ============== =====================================================
    tflag          '0' means the Verification tag is normal. '1' means
                   the Verification tag is copy of the sender.
    length         length of this chunk containing this header.
                   (0 means automatically-calculate when encoding)
    causes         a list of derived classes of os_ken.lib.packet.sctp.causes.
    ============== =====================================================
    """
    _class_prefixes = ['cause_']
    _RECOGNIZED_CAUSES = {}

    @staticmethod
    def register_cause_code(*args):

        def _register_cause_code(cls):
            chunk_abort._RECOGNIZED_CAUSES[cls.cause_code()] = cls
            return cls
        return _register_cause_code(args[0])

    @classmethod
    def chunk_type(cls):
        return TYPE_ABORT

    def __init__(self, tflag=0, length=0, causes=None):
        super(chunk_abort, self).__init__(self.chunk_type(), length)
        assert 1 == tflag | 1
        self.tflag = tflag
        causes = causes or []
        assert isinstance(causes, list)
        for one in causes:
            assert isinstance(one, cause)
        self.causes = causes

    @classmethod
    def parser(cls, buf):
        _, flags, length = struct.unpack_from(cls._PACK_STR, buf)
        tflag = flags >> 0 & 1
        causes = []
        offset = cls._MIN_LEN
        while offset < length:
            ccode, = struct.unpack_from('!H', buf, offset)
            cls_ = cls._RECOGNIZED_CAUSES.get(ccode)
            if not cls_:
                break
            ins = cls_.parser(buf[offset:])
            causes.append(ins)
            offset += len(ins)
        return cls(tflag, length, causes)

    def serialize(self):
        flags = self.tflag << 0
        buf = bytearray(struct.pack(self._PACK_STR, self.chunk_type(), flags, self.length))
        for one in self.causes:
            buf.extend(one.serialize())
        if 0 == self.length:
            self.length = len(buf)
            struct.pack_into('!H', buf, 2, self.length)
        return bytes(buf)