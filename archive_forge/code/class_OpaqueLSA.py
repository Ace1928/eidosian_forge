from functools import reduce
import logging
import struct
from os_ken.lib import addrconv
from os_ken.lib.packet import packet_base
from os_ken.lib.packet import packet_utils
from os_ken.lib.packet import stream_parser
from os_ken.lib.stringify import StringifyMixin
from os_ken.lib import type_desc
class OpaqueLSA(LSA):

    def __init__(self, data, *args, **kwargs):
        super(OpaqueLSA, self).__init__(*args, **kwargs)
        self.data = data

    @classmethod
    def parser(cls, buf, opaque_type=OSPF_OPAQUE_TYPE_UNKNOWN):
        opaquecls = OpaqueBody._lookup_type(opaque_type)
        if opaquecls:
            data = opaquecls.parser(buf)
        else:
            data = buf
        return {'data': data}

    def serialize_tail(self):
        if isinstance(self.data, OpaqueBody):
            return self.data.serialize()
        else:
            return self.data