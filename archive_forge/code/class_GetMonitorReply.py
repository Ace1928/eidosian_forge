import xcffib
import struct
import io
class GetMonitorReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.vendor_length, self.model_length, self.num_hsync, self.num_vsync = unpacker.unpack('xx2x4xBBBB20x')
        self.hsync = xcffib.List(unpacker, 'I', self.num_hsync)
        unpacker.pad('I')
        self.vsync = xcffib.List(unpacker, 'I', self.num_vsync)
        unpacker.pad('c')
        self.vendor = xcffib.List(unpacker, 'c', self.vendor_length)
        unpacker.pad('c')
        self.alignment_pad = xcffib.List(unpacker, 'c', (self.vendor_length + 3 & ~3) - self.vendor_length)
        unpacker.pad('c')
        self.model = xcffib.List(unpacker, 'c', self.model_length)
        self.bufsize = unpacker.offset - base