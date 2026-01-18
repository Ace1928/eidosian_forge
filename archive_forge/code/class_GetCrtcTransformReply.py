import xcffib
import struct
import io
from . import xproto
from . import render
class GetCrtcTransformReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        unpacker.unpack('xx2x4x')
        self.pending_transform = render.TRANSFORM(unpacker)
        self.has_transforms, = unpacker.unpack('B3x')
        unpacker.pad(render.TRANSFORM)
        self.current_transform = render.TRANSFORM(unpacker)
        self.pending_len, self.pending_nparams, self.current_len, self.current_nparams = unpacker.unpack('4xHHHH')
        unpacker.pad('c')
        self.pending_filter_name = xcffib.List(unpacker, 'c', self.pending_len)
        unpacker.pad('i')
        self.pending_params = xcffib.List(unpacker, 'i', self.pending_nparams)
        unpacker.pad('c')
        self.current_filter_name = xcffib.List(unpacker, 'c', self.current_len)
        unpacker.pad('i')
        self.current_params = xcffib.List(unpacker, 'i', self.current_nparams)
        self.bufsize = unpacker.offset - base