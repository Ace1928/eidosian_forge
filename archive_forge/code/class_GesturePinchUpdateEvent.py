import xcffib
import struct
import io
from . import xfixes
from . import xproto
class GesturePinchUpdateEvent(xcffib.Event):
    xge = True

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Event.__init__(self, unpacker)
        base = unpacker.offset
        self.deviceid, self.time, self.detail, self.root, self.event, self.child, self.root_x, self.root_y, self.event_x, self.event_y, self.delta_x, self.delta_y, self.delta_unaccel_x, self.delta_unaccel_y, self.scale, self.delta_angle, self.sourceid = unpacker.unpack('xx2xHIIIIIiiiiiiiiiiH2x')
        self.mods = ModifierInfo(unpacker)
        unpacker.pad(GroupInfo)
        self.group = GroupInfo(unpacker)
        self.flags, = unpacker.unpack('I')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=B', 28))
        buf.write(struct.pack('=x2xHIIIIIiiiiiiiiiiH2x', self.deviceid, self.time, self.detail, self.root, self.event, self.child, self.root_x, self.root_y, self.event_x, self.event_y, self.delta_x, self.delta_y, self.delta_unaccel_x, self.delta_unaccel_y, self.scale, self.delta_angle, self.sourceid))
        buf.write(self.mods.pack() if hasattr(self.mods, 'pack') else ModifierInfo.synthetic(*self.mods).pack())
        buf.write(self.group.pack() if hasattr(self.group, 'pack') else GroupInfo.synthetic(*self.group).pack())
        buf.write(struct.pack('=I', self.flags))
        buf_len = len(buf.getvalue())
        if buf_len < 32:
            buf.write(struct.pack('x' * (32 - buf_len)))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, deviceid, time, detail, root, event, child, root_x, root_y, event_x, event_y, delta_x, delta_y, delta_unaccel_x, delta_unaccel_y, scale, delta_angle, sourceid, mods, group, flags):
        self = cls.__new__(cls)
        self.deviceid = deviceid
        self.time = time
        self.detail = detail
        self.root = root
        self.event = event
        self.child = child
        self.root_x = root_x
        self.root_y = root_y
        self.event_x = event_x
        self.event_y = event_y
        self.delta_x = delta_x
        self.delta_y = delta_y
        self.delta_unaccel_x = delta_unaccel_x
        self.delta_unaccel_y = delta_unaccel_y
        self.scale = scale
        self.delta_angle = delta_angle
        self.sourceid = sourceid
        self.mods = mods
        self.group = group
        self.flags = flags
        return self