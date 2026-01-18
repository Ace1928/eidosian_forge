import xcffib
import struct
import io
from . import xproto
class xineramaExtension(xcffib.Extension):

    def QueryVersion(self, major, minor, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xBB', major, minor))
        return self.send_request(0, buf, QueryVersionCookie, is_checked=is_checked)

    def GetState(self, window, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', window))
        return self.send_request(1, buf, GetStateCookie, is_checked=is_checked)

    def GetScreenCount(self, window, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', window))
        return self.send_request(2, buf, GetScreenCountCookie, is_checked=is_checked)

    def GetScreenSize(self, window, screen, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', window, screen))
        return self.send_request(3, buf, GetScreenSizeCookie, is_checked=is_checked)

    def IsActive(self, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2x'))
        return self.send_request(4, buf, IsActiveCookie, is_checked=is_checked)

    def QueryScreens(self, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2x'))
        return self.send_request(5, buf, QueryScreensCookie, is_checked=is_checked)