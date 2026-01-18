import xcffib
import struct
import io
class xf86driExtension(xcffib.Extension):

    def QueryVersion(self, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2x'))
        return self.send_request(0, buf, QueryVersionCookie, is_checked=is_checked)

    def QueryDirectRenderingCapable(self, screen, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', screen))
        return self.send_request(1, buf, QueryDirectRenderingCapableCookie, is_checked=is_checked)

    def OpenConnection(self, screen, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', screen))
        return self.send_request(2, buf, OpenConnectionCookie, is_checked=is_checked)

    def CloseConnection(self, screen, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', screen))
        return self.send_request(3, buf, is_checked=is_checked)

    def GetClientDriverName(self, screen, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', screen))
        return self.send_request(4, buf, GetClientDriverNameCookie, is_checked=is_checked)

    def CreateContext(self, screen, visual, context, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIII', screen, visual, context))
        return self.send_request(5, buf, CreateContextCookie, is_checked=is_checked)

    def DestroyContext(self, screen, context, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', screen, context))
        return self.send_request(6, buf, is_checked=is_checked)

    def CreateDrawable(self, screen, drawable, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', screen, drawable))
        return self.send_request(7, buf, CreateDrawableCookie, is_checked=is_checked)

    def DestroyDrawable(self, screen, drawable, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', screen, drawable))
        return self.send_request(8, buf, is_checked=is_checked)

    def GetDrawableInfo(self, screen, drawable, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', screen, drawable))
        return self.send_request(9, buf, GetDrawableInfoCookie, is_checked=is_checked)

    def GetDeviceInfo(self, screen, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', screen))
        return self.send_request(10, buf, GetDeviceInfoCookie, is_checked=is_checked)

    def AuthConnection(self, screen, magic, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', screen, magic))
        return self.send_request(11, buf, AuthConnectionCookie, is_checked=is_checked)