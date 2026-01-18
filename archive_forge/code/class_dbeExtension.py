import xcffib
import struct
import io
from . import xproto
class dbeExtension(xcffib.Extension):

    def QueryVersion(self, major_version, minor_version, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xBB2x', major_version, minor_version))
        return self.send_request(0, buf, QueryVersionCookie, is_checked=is_checked)

    def AllocateBackBuffer(self, window, buffer, swap_action, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIB3x', window, buffer, swap_action))
        return self.send_request(1, buf, is_checked=is_checked)

    def DeallocateBackBuffer(self, buffer, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', buffer))
        return self.send_request(2, buf, is_checked=is_checked)

    def SwapBuffers(self, n_actions, actions, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', n_actions))
        buf.write(xcffib.pack_list(actions, SwapInfo))
        return self.send_request(3, buf, is_checked=is_checked)

    def BeginIdiom(self, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2x'))
        return self.send_request(4, buf, is_checked=is_checked)

    def EndIdiom(self, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2x'))
        return self.send_request(5, buf, is_checked=is_checked)

    def GetVisualInfo(self, n_drawables, drawables, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', n_drawables))
        buf.write(xcffib.pack_list(drawables, 'I'))
        return self.send_request(6, buf, GetVisualInfoCookie, is_checked=is_checked)

    def GetBackBufferAttributes(self, buffer, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', buffer))
        return self.send_request(7, buf, GetBackBufferAttributesCookie, is_checked=is_checked)