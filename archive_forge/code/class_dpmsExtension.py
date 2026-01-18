import xcffib
import struct
import io
from . import xproto
class dpmsExtension(xcffib.Extension):

    def GetVersion(self, client_major_version, client_minor_version, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xHH', client_major_version, client_minor_version))
        return self.send_request(0, buf, GetVersionCookie, is_checked=is_checked)

    def Capable(self, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2x'))
        return self.send_request(1, buf, CapableCookie, is_checked=is_checked)

    def GetTimeouts(self, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2x'))
        return self.send_request(2, buf, GetTimeoutsCookie, is_checked=is_checked)

    def SetTimeouts(self, standby_timeout, suspend_timeout, off_timeout, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xHHH', standby_timeout, suspend_timeout, off_timeout))
        return self.send_request(3, buf, is_checked=is_checked)

    def Enable(self, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2x'))
        return self.send_request(4, buf, is_checked=is_checked)

    def Disable(self, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2x'))
        return self.send_request(5, buf, is_checked=is_checked)

    def ForceLevel(self, power_level, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xH', power_level))
        return self.send_request(6, buf, is_checked=is_checked)

    def Info(self, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2x'))
        return self.send_request(7, buf, InfoCookie, is_checked=is_checked)

    def SelectInput(self, event_mask, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', event_mask))
        return self.send_request(8, buf, is_checked=is_checked)