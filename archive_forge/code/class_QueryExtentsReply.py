import xcffib
import struct
import io
from . import xproto
class QueryExtentsReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.bounding_shaped, self.clip_shaped, self.bounding_shape_extents_x, self.bounding_shape_extents_y, self.bounding_shape_extents_width, self.bounding_shape_extents_height, self.clip_shape_extents_x, self.clip_shape_extents_y, self.clip_shape_extents_width, self.clip_shape_extents_height = unpacker.unpack('xx2x4xBB2xhhHHhhHH')
        self.bufsize = unpacker.offset - base