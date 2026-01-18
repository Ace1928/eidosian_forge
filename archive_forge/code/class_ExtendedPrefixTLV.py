from functools import reduce
import logging
import struct
from os_ken.lib import addrconv
from os_ken.lib.packet import packet_base
from os_ken.lib.packet import packet_utils
from os_ken.lib.packet import stream_parser
from os_ken.lib.stringify import StringifyMixin
from os_ken.lib import type_desc
@ExtendedPrefixTLV.register_type(OSPF_EXTENDED_PREFIX_TLV)
class ExtendedPrefixTLV(ExtendedPrefixTLV):
    _VALUE_PACK_STR = '!HHBBBB4s'
    _VALUE_PACK_LEN = struct.calcsize(_VALUE_PACK_STR)
    _VALUE_FIELDS = ['route_type', 'prefix_length', 'address_family', '_padprefix']

    def __init__(self, type_=OSPF_EXTENDED_PREFIX_TLV, length=0, route_type=0, address_family=0, prefix='0.0.0.0/0'):
        self.type_ = type_
        self.length = length
        self.route_type = route_type
        self.address_family = address_family
        self.prefix = prefix

    @classmethod
    def parser(cls, buf):
        rest = buf[cls._VALUE_PACK_LEN:]
        buf = buf[:cls._VALUE_PACK_LEN]
        type_, length, route_type, prefix_length, address_family, _pad, prefix = struct.unpack_from(cls._VALUE_PACK_STR, buf)
        prefix = addrconv.ipv4.bin_to_text(prefix)
        prefix = '%s/%d' % (prefix, prefix_length)
        return (cls(type_, length, route_type, address_family, prefix), rest)

    def serialize(self):
        prefix, prefix_length = self.prefix.split('/')
        prefix = addrconv.ipv4.text_to_bin(prefix)
        prefix_length = int(prefix_length)
        return struct.pack(self._VALUE_PACK_STR, OSPF_EXTENDED_PREFIX_TLV, self._VALUE_PACK_LEN - 4, self.route_type, prefix_length, self.address_family, 0, prefix)