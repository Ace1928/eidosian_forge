import struct
import logging
from os_ken.lib import stringify
from . import packet_base
from . import packet_utils
from . import bgp
from . import openflow
from . import zebra
@TCPOption.register(TCP_OPTION_KIND_SACK, 2)
class TCPOptionSACK(TCPOption):
    _PACK_STR = '!BB'
    _BLOCK_PACK_STR = '!II'

    def __init__(self, blocks, kind=None, length=None):
        super(TCPOptionSACK, self).__init__(kind, length)
        self.blocks = blocks

    @classmethod
    def parse(cls, buf):
        _, length = struct.unpack_from(cls._PACK_STR, buf)
        blocks_buf = buf[2:length]
        blocks = []
        while blocks_buf:
            lr_block = struct.unpack_from(cls._BLOCK_PACK_STR, blocks_buf)
            blocks.append(lr_block)
            blocks_buf = blocks_buf[8:]
        return (cls(blocks, cls.cls_kind, length), buf[length:])

    def serialize(self):
        buf = bytearray()
        for left, right in self.blocks:
            buf += struct.pack(self._BLOCK_PACK_STR, left, right)
        self.length = self.cls_length + len(buf)
        return struct.pack(self._PACK_STR, self.kind, self.length) + buf