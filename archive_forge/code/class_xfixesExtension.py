import xcffib
import struct
import io
from . import xproto
from . import render
from . import shape
class xfixesExtension(xcffib.Extension):

    def QueryVersion(self, client_major_version, client_minor_version, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', client_major_version, client_minor_version))
        return self.send_request(0, buf, QueryVersionCookie, is_checked=is_checked)

    def ChangeSaveSet(self, mode, target, map, window, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xBBBxI', mode, target, map, window))
        return self.send_request(1, buf, is_checked=is_checked)

    def SelectSelectionInput(self, window, selection, event_mask, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIII', window, selection, event_mask))
        return self.send_request(2, buf, is_checked=is_checked)

    def SelectCursorInput(self, window, event_mask, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', window, event_mask))
        return self.send_request(3, buf, is_checked=is_checked)

    def GetCursorImage(self, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2x'))
        return self.send_request(4, buf, GetCursorImageCookie, is_checked=is_checked)

    def CreateRegion(self, region, rectangles_len, rectangles, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', region))
        buf.write(xcffib.pack_list(rectangles, xproto.RECTANGLE))
        return self.send_request(5, buf, is_checked=is_checked)

    def CreateRegionFromBitmap(self, region, bitmap, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', region, bitmap))
        return self.send_request(6, buf, is_checked=is_checked)

    def CreateRegionFromWindow(self, region, window, kind, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIB3x', region, window, kind))
        return self.send_request(7, buf, is_checked=is_checked)

    def CreateRegionFromGC(self, region, gc, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', region, gc))
        return self.send_request(8, buf, is_checked=is_checked)

    def CreateRegionFromPicture(self, region, picture, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', region, picture))
        return self.send_request(9, buf, is_checked=is_checked)

    def DestroyRegion(self, region, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', region))
        return self.send_request(10, buf, is_checked=is_checked)

    def SetRegion(self, region, rectangles_len, rectangles, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', region))
        buf.write(xcffib.pack_list(rectangles, xproto.RECTANGLE))
        return self.send_request(11, buf, is_checked=is_checked)

    def CopyRegion(self, source, destination, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', source, destination))
        return self.send_request(12, buf, is_checked=is_checked)

    def UnionRegion(self, source1, source2, destination, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIII', source1, source2, destination))
        return self.send_request(13, buf, is_checked=is_checked)

    def IntersectRegion(self, source1, source2, destination, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIII', source1, source2, destination))
        return self.send_request(14, buf, is_checked=is_checked)

    def SubtractRegion(self, source1, source2, destination, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIII', source1, source2, destination))
        return self.send_request(15, buf, is_checked=is_checked)

    def InvertRegion(self, source, bounds, destination, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', source))
        buf.write(bounds.pack() if hasattr(bounds, 'pack') else xproto.RECTANGLE.synthetic(*bounds).pack())
        buf.write(struct.pack('=I', destination))
        return self.send_request(16, buf, is_checked=is_checked)

    def TranslateRegion(self, region, dx, dy, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIhh', region, dx, dy))
        return self.send_request(17, buf, is_checked=is_checked)

    def RegionExtents(self, source, destination, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', source, destination))
        return self.send_request(18, buf, is_checked=is_checked)

    def FetchRegion(self, region, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', region))
        return self.send_request(19, buf, FetchRegionCookie, is_checked=is_checked)

    def SetGCClipRegion(self, gc, region, x_origin, y_origin, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIhh', gc, region, x_origin, y_origin))
        return self.send_request(20, buf, is_checked=is_checked)

    def SetWindowShapeRegion(self, dest, dest_kind, x_offset, y_offset, region, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIB3xhhI', dest, dest_kind, x_offset, y_offset, region))
        return self.send_request(21, buf, is_checked=is_checked)

    def SetPictureClipRegion(self, picture, region, x_origin, y_origin, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIhh', picture, region, x_origin, y_origin))
        return self.send_request(22, buf, is_checked=is_checked)

    def SetCursorName(self, cursor, nbytes, name, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIH2x', cursor, nbytes))
        buf.write(xcffib.pack_list(name, 'c'))
        return self.send_request(23, buf, is_checked=is_checked)

    def GetCursorName(self, cursor, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', cursor))
        return self.send_request(24, buf, GetCursorNameCookie, is_checked=is_checked)

    def GetCursorImageAndName(self, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2x'))
        return self.send_request(25, buf, GetCursorImageAndNameCookie, is_checked=is_checked)

    def ChangeCursor(self, source, destination, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', source, destination))
        return self.send_request(26, buf, is_checked=is_checked)

    def ChangeCursorByName(self, src, nbytes, name, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIH2x', src, nbytes))
        buf.write(xcffib.pack_list(name, 'c'))
        return self.send_request(27, buf, is_checked=is_checked)

    def ExpandRegion(self, source, destination, left, right, top, bottom, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIHHHH', source, destination, left, right, top, bottom))
        return self.send_request(28, buf, is_checked=is_checked)

    def HideCursor(self, window, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', window))
        return self.send_request(29, buf, is_checked=is_checked)

    def ShowCursor(self, window, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', window))
        return self.send_request(30, buf, is_checked=is_checked)

    def CreatePointerBarrier(self, barrier, window, x1, y1, x2, y2, directions, num_devices, devices, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIHHHHI2xH', barrier, window, x1, y1, x2, y2, directions, num_devices))
        buf.write(xcffib.pack_list(devices, 'H'))
        return self.send_request(31, buf, is_checked=is_checked)

    def DeletePointerBarrier(self, barrier, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', barrier))
        return self.send_request(32, buf, is_checked=is_checked)

    def SetClientDisconnectMode(self, disconnect_mode, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', disconnect_mode))
        return self.send_request(33, buf, is_checked=is_checked)

    def GetClientDisconnectMode(self, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2x'))
        return self.send_request(34, buf, GetClientDisconnectModeCookie, is_checked=is_checked)