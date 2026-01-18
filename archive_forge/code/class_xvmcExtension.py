import xcffib
import struct
import io
from . import xv
class xvmcExtension(xcffib.Extension):

    def QueryVersion(self, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2x'))
        return self.send_request(0, buf, QueryVersionCookie, is_checked=is_checked)

    def ListSurfaceTypes(self, port_id, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', port_id))
        return self.send_request(1, buf, ListSurfaceTypesCookie, is_checked=is_checked)

    def CreateContext(self, context_id, port_id, surface_id, width, height, flags, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIIHHI', context_id, port_id, surface_id, width, height, flags))
        return self.send_request(2, buf, CreateContextCookie, is_checked=is_checked)

    def DestroyContext(self, context_id, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', context_id))
        return self.send_request(3, buf, is_checked=is_checked)

    def CreateSurface(self, surface_id, context_id, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', surface_id, context_id))
        return self.send_request(4, buf, CreateSurfaceCookie, is_checked=is_checked)

    def DestroySurface(self, surface_id, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', surface_id))
        return self.send_request(5, buf, is_checked=is_checked)

    def CreateSubpicture(self, subpicture_id, context, xvimage_id, width, height, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIIHH', subpicture_id, context, xvimage_id, width, height))
        return self.send_request(6, buf, CreateSubpictureCookie, is_checked=is_checked)

    def DestroySubpicture(self, subpicture_id, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', subpicture_id))
        return self.send_request(7, buf, is_checked=is_checked)

    def ListSubpictureTypes(self, port_id, surface_id, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', port_id, surface_id))
        return self.send_request(8, buf, ListSubpictureTypesCookie, is_checked=is_checked)