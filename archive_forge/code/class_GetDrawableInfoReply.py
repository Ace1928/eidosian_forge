import xcffib
import struct
import io
class GetDrawableInfoReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.drawable_table_index, self.drawable_table_stamp, self.drawable_origin_X, self.drawable_origin_Y, self.drawable_size_W, self.drawable_size_H, self.num_clip_rects, self.back_x, self.back_y, self.num_back_clip_rects = unpacker.unpack('xx2x4xIIhhhhIhhI')
        self.clip_rects = xcffib.List(unpacker, DrmClipRect, self.num_clip_rects)
        unpacker.pad(DrmClipRect)
        self.back_clip_rects = xcffib.List(unpacker, DrmClipRect, self.num_back_clip_rects)
        self.bufsize = unpacker.offset - base