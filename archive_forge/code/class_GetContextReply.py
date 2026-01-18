import xcffib
import struct
import io
class GetContextReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.enabled, self.element_header, self.num_intercepted_clients = unpacker.unpack('xB2x4xB3xI16x')
        self.intercepted_clients = xcffib.List(unpacker, ClientInfo, self.num_intercepted_clients)
        self.bufsize = unpacker.offset - base