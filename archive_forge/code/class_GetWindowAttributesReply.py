import xcffib
import struct
import io
class GetWindowAttributesReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.backing_store, self.visual, self._class, self.bit_gravity, self.win_gravity, self.backing_planes, self.backing_pixel, self.save_under, self.map_is_installed, self.map_state, self.override_redirect, self.colormap, self.all_event_masks, self.your_event_mask, self.do_not_propagate_mask = unpacker.unpack('xB2x4xIHBBIIBBBBIIIH2x')
        self.bufsize = unpacker.offset - base