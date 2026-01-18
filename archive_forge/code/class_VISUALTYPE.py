import xcffib
import struct
import io
class VISUALTYPE(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.visual_id, self._class, self.bits_per_rgb_value, self.colormap_entries, self.red_mask, self.green_mask, self.blue_mask = unpacker.unpack('IBBHIII4x')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=IBBHIII4x', self.visual_id, self._class, self.bits_per_rgb_value, self.colormap_entries, self.red_mask, self.green_mask, self.blue_mask))
        return buf.getvalue()
    fixed_size = 24

    @classmethod
    def synthetic(cls, visual_id, _class, bits_per_rgb_value, colormap_entries, red_mask, green_mask, blue_mask):
        self = cls.__new__(cls)
        self.visual_id = visual_id
        self._class = _class
        self.bits_per_rgb_value = bits_per_rgb_value
        self.colormap_entries = colormap_entries
        self.red_mask = red_mask
        self.green_mask = green_mask
        self.blue_mask = blue_mask
        return self