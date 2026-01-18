from functools import reduce
import logging
import struct
from os_ken.lib import addrconv
from os_ken.lib.packet import packet_base
from os_ken.lib.packet import packet_utils
from os_ken.lib.packet import stream_parser
from os_ken.lib.stringify import StringifyMixin
from os_ken.lib import type_desc
@LSA.register_type(OSPF_ROUTER_LSA)
class RouterLSA(LSA):
    _PACK_STR = '!BBH'
    _PACK_LEN = struct.calcsize(_PACK_STR)

    class Link(StringifyMixin):
        _PACK_STR = '!4s4sBBH'
        _PACK_LEN = struct.calcsize(_PACK_STR)

        def __init__(self, id_='0.0.0.0', data='0.0.0.0', type_=LSA_LINK_TYPE_STUB, tos=0, metric=10):
            self.id_ = id_
            self.data = data
            self.type_ = type_
            self.tos = tos
            self.metric = metric

        @classmethod
        def parser(cls, buf):
            if len(buf) < cls._PACK_LEN:
                raise stream_parser.StreamParser.TooSmallException('%d < %d' % (len(buf), cls._PACK_LEN))
            link = buf[:cls._PACK_LEN]
            rest = buf[cls._PACK_LEN:]
            id_, data, type_, tos, metric = struct.unpack_from(cls._PACK_STR, bytes(link))
            id_ = addrconv.ipv4.bin_to_text(id_)
            data = addrconv.ipv4.bin_to_text(data)
            return (cls(id_, data, type_, tos, metric), rest)

        def serialize(self):
            id_ = addrconv.ipv4.text_to_bin(self.id_)
            data = addrconv.ipv4.text_to_bin(self.data)
            return bytearray(struct.pack(self._PACK_STR, id_, data, self.type_, self.tos, self.metric))

    def __init__(self, ls_age=0, options=0, type_=OSPF_ROUTER_LSA, id_='0.0.0.0', adv_router='0.0.0.0', ls_seqnum=0, checksum=None, length=None, flags=0, links=None):
        links = links if links else []
        self.flags = flags
        self.links = links
        super(RouterLSA, self).__init__(ls_age, options, type_, id_, adv_router, ls_seqnum, checksum, length)

    @classmethod
    def parser(cls, buf):
        links = []
        hdr = buf[:cls._PACK_LEN]
        buf = buf[cls._PACK_LEN:]
        flags, _, num = struct.unpack_from(cls._PACK_STR, bytes(hdr))
        while buf:
            link, buf = cls.Link.parser(buf)
            links.append(link)
        assert len(links) == num
        return {'flags': flags, 'links': links}

    def serialize_tail(self):
        head = bytearray(struct.pack(self._PACK_STR, self.flags, 0, len(self.links)))
        try:
            return head + reduce(lambda a, b: a + b, (link.serialize() for link in self.links))
        except TypeError:
            return head