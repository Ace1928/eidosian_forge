import xcffib
import struct
import io
from . import xproto
def ChangePicture(self, picture, value_mask, value_list, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xII', picture, value_mask))
    if value_mask & CP.Repeat:
        repeat = value_list.pop(0)
        buf.write(struct.pack('=I', repeat))
    if value_mask & CP.AlphaMap:
        alphamap = value_list.pop(0)
        buf.write(struct.pack('=I', alphamap))
    if value_mask & CP.AlphaXOrigin:
        alphaxorigin = value_list.pop(0)
        buf.write(struct.pack('=i', alphaxorigin))
    if value_mask & CP.AlphaYOrigin:
        alphayorigin = value_list.pop(0)
        buf.write(struct.pack('=i', alphayorigin))
    if value_mask & CP.ClipXOrigin:
        clipxorigin = value_list.pop(0)
        buf.write(struct.pack('=i', clipxorigin))
    if value_mask & CP.ClipYOrigin:
        clipyorigin = value_list.pop(0)
        buf.write(struct.pack('=i', clipyorigin))
    if value_mask & CP.ClipMask:
        clipmask = value_list.pop(0)
        buf.write(struct.pack('=I', clipmask))
    if value_mask & CP.GraphicsExposure:
        graphicsexposure = value_list.pop(0)
        buf.write(struct.pack('=I', graphicsexposure))
    if value_mask & CP.SubwindowMode:
        subwindowmode = value_list.pop(0)
        buf.write(struct.pack('=I', subwindowmode))
    if value_mask & CP.PolyEdge:
        polyedge = value_list.pop(0)
        buf.write(struct.pack('=I', polyedge))
    if value_mask & CP.PolyMode:
        polymode = value_list.pop(0)
        buf.write(struct.pack('=I', polymode))
    if value_mask & CP.Dither:
        dither = value_list.pop(0)
        buf.write(struct.pack('=I', dither))
    if value_mask & CP.ComponentAlpha:
        componentalpha = value_list.pop(0)
        buf.write(struct.pack('=I', componentalpha))
    return self.send_request(5, buf, is_checked=is_checked)