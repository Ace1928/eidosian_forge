from functools import reduce
import logging
import struct
from os_ken.lib import addrconv
from os_ken.lib.packet import packet_base
from os_ken.lib.packet import packet_utils
from os_ken.lib.packet import stream_parser
from os_ken.lib.stringify import StringifyMixin
from os_ken.lib import type_desc
@OSPFMessage.register_type(OSPF_MSG_HELLO)
class OSPFHello(OSPFMessage):
    _PACK_STR = '!4sHBBI4s4s'
    _PACK_LEN = struct.calcsize(_PACK_STR)
    _MIN_LEN = OSPFMessage._HDR_LEN + _PACK_LEN

    def __init__(self, length=None, router_id='0.0.0.0', area_id='0.0.0.0', au_type=1, authentication=0, checksum=None, version=_VERSION, mask='0.0.0.0', hello_interval=10, options=0, priority=1, dead_interval=40, designated_router='0.0.0.0', backup_router='0.0.0.0', neighbors=None):
        neighbors = neighbors if neighbors else []
        super(OSPFHello, self).__init__(OSPF_MSG_HELLO, length, router_id, area_id, au_type, authentication, checksum, version)
        self.mask = mask
        self.hello_interval = hello_interval
        self.options = options
        self.priority = priority
        self.dead_interval = dead_interval
        self.designated_router = designated_router
        self.backup_router = backup_router
        self.neighbors = neighbors

    @classmethod
    def parser(cls, buf):
        mask, hello_interval, options, priority, dead_interval, designated_router, backup_router = struct.unpack_from(cls._PACK_STR, bytes(buf))
        mask = addrconv.ipv4.bin_to_text(mask)
        designated_router = addrconv.ipv4.bin_to_text(designated_router)
        backup_router = addrconv.ipv4.bin_to_text(backup_router)
        neighbors = []
        binneighbors = buf[cls._PACK_LEN:len(buf)]
        while binneighbors:
            n = binneighbors[:4]
            n = addrconv.ipv4.bin_to_text(bytes(n))
            binneighbors = binneighbors[4:]
            neighbors.append(n)
        return {'mask': mask, 'hello_interval': hello_interval, 'options': options, 'priority': priority, 'dead_interval': dead_interval, 'designated_router': designated_router, 'backup_router': backup_router, 'neighbors': neighbors}

    def serialize_tail(self):
        head = bytearray(struct.pack(self._PACK_STR, addrconv.ipv4.text_to_bin(self.mask), self.hello_interval, self.options, self.priority, self.dead_interval, addrconv.ipv4.text_to_bin(self.designated_router), addrconv.ipv4.text_to_bin(self.backup_router)))
        try:
            return head + reduce(lambda a, b: a + b, (addrconv.ipv4.text_to_bin(n) for n in self.neighbors))
        except TypeError:
            return head