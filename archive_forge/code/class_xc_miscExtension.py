import xcffib
import struct
import io
class xc_miscExtension(xcffib.Extension):

    def GetVersion(self, client_major_version, client_minor_version, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xHH', client_major_version, client_minor_version))
        return self.send_request(0, buf, GetVersionCookie, is_checked=is_checked)

    def GetXIDRange(self, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2x'))
        return self.send_request(1, buf, GetXIDRangeCookie, is_checked=is_checked)

    def GetXIDList(self, count, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', count))
        return self.send_request(2, buf, GetXIDListCookie, is_checked=is_checked)