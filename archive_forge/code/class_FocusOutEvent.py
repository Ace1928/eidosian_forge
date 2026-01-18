import xcffib
import struct
import io
from . import xfixes
from . import xproto
class FocusOutEvent(xcffib.Event):
    xge = True

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Event.__init__(self, unpacker)
        base = unpacker.offset
        self.deviceid, self.time, self.sourceid, self.mode, self.detail, self.root, self.event, self.child, self.root_x, self.root_y, self.event_x, self.event_y, self.same_screen, self.focus, self.buttons_len = unpacker.unpack('xx2xHIHBBIIIiiiiBBH')
        self.mods = ModifierInfo(unpacker)
        unpacker.pad(GroupInfo)
        self.group = GroupInfo(unpacker)
        unpacker.pad('I')
        self.buttons = xcffib.List(unpacker, 'I', self.buttons_len)
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=B', 10))
        buf.write(struct.pack('=x2xHIHBBIIIiiiiBBH', self.deviceid, self.time, self.sourceid, self.mode, self.detail, self.root, self.event, self.child, self.root_x, self.root_y, self.event_x, self.event_y, self.same_screen, self.focus, self.buttons_len))
        buf.write(self.mods.pack() if hasattr(self.mods, 'pack') else ModifierInfo.synthetic(*self.mods).pack())
        buf.write(self.group.pack() if hasattr(self.group, 'pack') else GroupInfo.synthetic(*self.group).pack())
        buf.write(xcffib.pack_list(self.buttons, 'I'))
        buf_len = len(buf.getvalue())
        if buf_len < 32:
            buf.write(struct.pack('x' * (32 - buf_len)))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, deviceid, time, sourceid, mode, detail, root, event, child, root_x, root_y, event_x, event_y, same_screen, focus, buttons_len, mods, group, buttons):
        self = cls.__new__(cls)
        self.deviceid = deviceid
        self.time = time
        self.sourceid = sourceid
        self.mode = mode
        self.detail = detail
        self.root = root
        self.event = event
        self.child = child
        self.root_x = root_x
        self.root_y = root_y
        self.event_x = event_x
        self.event_y = event_y
        self.same_screen = same_screen
        self.focus = focus
        self.buttons_len = buttons_len
        self.mods = mods
        self.group = group
        self.buttons = buttons
        return self