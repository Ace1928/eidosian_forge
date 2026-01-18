import abc
import logging
import struct
import time
from os_ken.lib import addrconv
from os_ken.lib import ip
from os_ken.lib import stringify
from os_ken.lib import type_desc
from os_ken.lib.packet import bgp
from os_ken.lib.packet import ospf
class MrtRecord(stringify.StringifyMixin, type_desc.TypeDisp, metaclass=abc.ABCMeta):
    """
    MRT record.
    """
    _HEADER_FMT = '!IHHI'
    HEADER_SIZE = struct.calcsize(_HEADER_FMT)
    MESSAGE_CLS = None
    TYPE_OSPFv2 = 11
    TYPE_TABLE_DUMP = 12
    TYPE_TABLE_DUMP_V2 = 13
    TYPE_BGP4MP = 16
    TYPE_BGP4MP_ET = 17
    TYPE_ISIS = 32
    TYPE_ISIS_ET = 33
    TYPE_OSPFv3 = 48
    TYPE_OSPFv3_ET = 49
    _EXT_TS_TYPES = [TYPE_BGP4MP_ET, TYPE_ISIS_ET, TYPE_OSPFv3_ET]

    def __init__(self, message, timestamp=None, type_=None, subtype=None, length=None):
        assert issubclass(message.__class__, MrtMessage)
        self.message = message
        self.timestamp = timestamp
        if type_ is None:
            type_ = self._rev_lookup_type(self.__class__)
        self.type = type_
        if subtype is None:
            subtype = self.MESSAGE_CLS._rev_lookup_type(message.__class__)
        self.subtype = subtype
        self.length = length

    @classmethod
    def parse_common_header(cls, buf):
        header_fields = struct.unpack_from(cls._HEADER_FMT, buf)
        return (list(header_fields), buf[cls.HEADER_SIZE:])

    @classmethod
    def parse_extended_header(cls, buf):
        return ([], buf)

    @classmethod
    def parse_pre(cls, buf):
        buf = bytes(buf)
        header_fields, _ = cls.parse_common_header(buf)
        type_ = header_fields[1]
        length = header_fields[3]
        if type_ in cls._EXT_TS_TYPES:
            header_cls = ExtendedTimestampMrtRecord
        else:
            header_cls = MrtCommonRecord
        required_len = header_cls.HEADER_SIZE + length
        return required_len

    @classmethod
    def parse(cls, buf):
        buf = bytes(buf)
        header_fields, rest = cls.parse_common_header(buf)
        type_ = header_fields[1]
        subtype = header_fields[2]
        length = header_fields[3]
        sub_cls = MrtRecord._lookup_type(type_)
        extended_headers, rest = sub_cls.parse_extended_header(rest)
        header_fields.extend(extended_headers)
        msg_cls = sub_cls.MESSAGE_CLS._lookup_type(subtype)
        message_bin = rest[:length]
        message = msg_cls.parse(message_bin)
        return (sub_cls(message, *header_fields), rest[length:])

    @abc.abstractmethod
    def serialize_header(self):
        pass

    def serialize(self):
        if self.timestamp is None:
            self.timestamp = int(time.time())
        buf = self.message.serialize()
        self.length = len(buf)
        return self.serialize_header() + buf