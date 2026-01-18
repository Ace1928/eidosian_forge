import xcffib
import struct
import io
class GetKeyboardControlReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.global_auto_repeat, self.led_mask, self.key_click_percent, self.bell_percent, self.bell_pitch, self.bell_duration = unpacker.unpack('xB2x4xIBBHH2x')
        self.auto_repeats = xcffib.List(unpacker, 'B', 32)
        self.bufsize = unpacker.offset - base