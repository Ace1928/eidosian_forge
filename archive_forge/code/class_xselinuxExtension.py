import xcffib
import struct
import io
from . import xproto
class xselinuxExtension(xcffib.Extension):

    def QueryVersion(self, client_major, client_minor, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xBB', client_major, client_minor))
        return self.send_request(0, buf, QueryVersionCookie, is_checked=is_checked)

    def SetDeviceCreateContext(self, context_len, context, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', context_len))
        buf.write(xcffib.pack_list(context, 'c'))
        return self.send_request(1, buf, is_checked=is_checked)

    def GetDeviceCreateContext(self, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2x'))
        return self.send_request(2, buf, GetDeviceCreateContextCookie, is_checked=is_checked)

    def SetDeviceContext(self, device, context_len, context, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', device, context_len))
        buf.write(xcffib.pack_list(context, 'c'))
        return self.send_request(3, buf, is_checked=is_checked)

    def GetDeviceContext(self, device, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', device))
        return self.send_request(4, buf, GetDeviceContextCookie, is_checked=is_checked)

    def SetWindowCreateContext(self, context_len, context, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', context_len))
        buf.write(xcffib.pack_list(context, 'c'))
        return self.send_request(5, buf, is_checked=is_checked)

    def GetWindowCreateContext(self, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2x'))
        return self.send_request(6, buf, GetWindowCreateContextCookie, is_checked=is_checked)

    def GetWindowContext(self, window, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', window))
        return self.send_request(7, buf, GetWindowContextCookie, is_checked=is_checked)

    def SetPropertyCreateContext(self, context_len, context, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', context_len))
        buf.write(xcffib.pack_list(context, 'c'))
        return self.send_request(8, buf, is_checked=is_checked)

    def GetPropertyCreateContext(self, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2x'))
        return self.send_request(9, buf, GetPropertyCreateContextCookie, is_checked=is_checked)

    def SetPropertyUseContext(self, context_len, context, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', context_len))
        buf.write(xcffib.pack_list(context, 'c'))
        return self.send_request(10, buf, is_checked=is_checked)

    def GetPropertyUseContext(self, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2x'))
        return self.send_request(11, buf, GetPropertyUseContextCookie, is_checked=is_checked)

    def GetPropertyContext(self, window, property, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', window, property))
        return self.send_request(12, buf, GetPropertyContextCookie, is_checked=is_checked)

    def GetPropertyDataContext(self, window, property, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', window, property))
        return self.send_request(13, buf, GetPropertyDataContextCookie, is_checked=is_checked)

    def ListProperties(self, window, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', window))
        return self.send_request(14, buf, ListPropertiesCookie, is_checked=is_checked)

    def SetSelectionCreateContext(self, context_len, context, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', context_len))
        buf.write(xcffib.pack_list(context, 'c'))
        return self.send_request(15, buf, is_checked=is_checked)

    def GetSelectionCreateContext(self, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2x'))
        return self.send_request(16, buf, GetSelectionCreateContextCookie, is_checked=is_checked)

    def SetSelectionUseContext(self, context_len, context, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', context_len))
        buf.write(xcffib.pack_list(context, 'c'))
        return self.send_request(17, buf, is_checked=is_checked)

    def GetSelectionUseContext(self, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2x'))
        return self.send_request(18, buf, GetSelectionUseContextCookie, is_checked=is_checked)

    def GetSelectionContext(self, selection, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', selection))
        return self.send_request(19, buf, GetSelectionContextCookie, is_checked=is_checked)

    def GetSelectionDataContext(self, selection, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', selection))
        return self.send_request(20, buf, GetSelectionDataContextCookie, is_checked=is_checked)

    def ListSelections(self, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2x'))
        return self.send_request(21, buf, ListSelectionsCookie, is_checked=is_checked)

    def GetClientContext(self, resource, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', resource))
        return self.send_request(22, buf, GetClientContextCookie, is_checked=is_checked)