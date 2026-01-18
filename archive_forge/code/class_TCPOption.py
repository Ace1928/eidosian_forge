import struct
import logging
from os_ken.lib import stringify
from . import packet_base
from . import packet_utils
from . import bgp
from . import openflow
from . import zebra
class TCPOption(stringify.StringifyMixin):
    _KINDS = {}
    _KIND_PACK_STR = '!B'
    NO_BODY_OFFSET = 1
    WITH_BODY_OFFSET = 2
    cls_kind = None
    cls_length = None

    def __init__(self, kind=None, length=None):
        self.kind = self.cls_kind if kind is None else kind
        self.length = self.cls_length if length is None else length

    @classmethod
    def register(cls, kind, length):

        def _register(subcls):
            subcls.cls_kind = kind
            subcls.cls_length = length
            cls._KINDS[kind] = subcls
            return subcls
        return _register

    @classmethod
    def parse(cls, buf):
        return (cls(cls.cls_kind, cls.cls_length), buf[cls.cls_length:])

    @classmethod
    def parser(cls, buf):
        kind, = struct.unpack_from(cls._KIND_PACK_STR, buf)
        subcls = cls._KINDS.get(kind)
        if not subcls:
            subcls = TCPOptionUnknown
        return subcls.parse(buf)

    def serialize(self):
        return struct.pack(self._KIND_PACK_STR, self.cls_kind)