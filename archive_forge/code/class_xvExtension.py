import xcffib
import struct
import io
from . import xproto
from . import shm
class xvExtension(xcffib.Extension):

    def QueryExtension(self, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2x'))
        return self.send_request(0, buf, QueryExtensionCookie, is_checked=is_checked)

    def QueryAdaptors(self, window, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', window))
        return self.send_request(1, buf, QueryAdaptorsCookie, is_checked=is_checked)

    def QueryEncodings(self, port, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', port))
        return self.send_request(2, buf, QueryEncodingsCookie, is_checked=is_checked)

    def GrabPort(self, port, time, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', port, time))
        return self.send_request(3, buf, GrabPortCookie, is_checked=is_checked)

    def UngrabPort(self, port, time, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', port, time))
        return self.send_request(4, buf, is_checked=is_checked)

    def PutVideo(self, port, drawable, gc, vid_x, vid_y, vid_w, vid_h, drw_x, drw_y, drw_w, drw_h, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIIhhHHhhHH', port, drawable, gc, vid_x, vid_y, vid_w, vid_h, drw_x, drw_y, drw_w, drw_h))
        return self.send_request(5, buf, is_checked=is_checked)

    def PutStill(self, port, drawable, gc, vid_x, vid_y, vid_w, vid_h, drw_x, drw_y, drw_w, drw_h, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIIhhHHhhHH', port, drawable, gc, vid_x, vid_y, vid_w, vid_h, drw_x, drw_y, drw_w, drw_h))
        return self.send_request(6, buf, is_checked=is_checked)

    def GetVideo(self, port, drawable, gc, vid_x, vid_y, vid_w, vid_h, drw_x, drw_y, drw_w, drw_h, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIIhhHHhhHH', port, drawable, gc, vid_x, vid_y, vid_w, vid_h, drw_x, drw_y, drw_w, drw_h))
        return self.send_request(7, buf, is_checked=is_checked)

    def GetStill(self, port, drawable, gc, vid_x, vid_y, vid_w, vid_h, drw_x, drw_y, drw_w, drw_h, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIIhhHHhhHH', port, drawable, gc, vid_x, vid_y, vid_w, vid_h, drw_x, drw_y, drw_w, drw_h))
        return self.send_request(8, buf, is_checked=is_checked)

    def StopVideo(self, port, drawable, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', port, drawable))
        return self.send_request(9, buf, is_checked=is_checked)

    def SelectVideoNotify(self, drawable, onoff, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIB3x', drawable, onoff))
        return self.send_request(10, buf, is_checked=is_checked)

    def SelectPortNotify(self, port, onoff, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIB3x', port, onoff))
        return self.send_request(11, buf, is_checked=is_checked)

    def QueryBestSize(self, port, vid_w, vid_h, drw_w, drw_h, motion, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIHHHHB3x', port, vid_w, vid_h, drw_w, drw_h, motion))
        return self.send_request(12, buf, QueryBestSizeCookie, is_checked=is_checked)

    def SetPortAttribute(self, port, attribute, value, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIi', port, attribute, value))
        return self.send_request(13, buf, is_checked=is_checked)

    def GetPortAttribute(self, port, attribute, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', port, attribute))
        return self.send_request(14, buf, GetPortAttributeCookie, is_checked=is_checked)

    def QueryPortAttributes(self, port, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', port))
        return self.send_request(15, buf, QueryPortAttributesCookie, is_checked=is_checked)

    def ListImageFormats(self, port, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', port))
        return self.send_request(16, buf, ListImageFormatsCookie, is_checked=is_checked)

    def QueryImageAttributes(self, port, id, width, height, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIHH', port, id, width, height))
        return self.send_request(17, buf, QueryImageAttributesCookie, is_checked=is_checked)

    def PutImage(self, port, drawable, gc, id, src_x, src_y, src_w, src_h, drw_x, drw_y, drw_w, drw_h, width, height, data_len, data, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIIIhhHHhhHHHH', port, drawable, gc, id, src_x, src_y, src_w, src_h, drw_x, drw_y, drw_w, drw_h, width, height))
        buf.write(xcffib.pack_list(data, 'B'))
        return self.send_request(18, buf, is_checked=is_checked)

    def ShmPutImage(self, port, drawable, gc, shmseg, id, offset, src_x, src_y, src_w, src_h, drw_x, drw_y, drw_w, drw_h, width, height, send_event, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIIIIIhhHHhhHHHHB3x', port, drawable, gc, shmseg, id, offset, src_x, src_y, src_w, src_h, drw_x, drw_y, drw_w, drw_h, width, height, send_event))
        return self.send_request(19, buf, is_checked=is_checked)