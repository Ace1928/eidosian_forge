from Xlib import X
from Xlib.protocol import rq, structs
class CopyColormapAndFree(rq.Request):
    _request = rq.Struct(rq.Opcode(80), rq.Pad(1), rq.RequestLength(), rq.Colormap('mid'), rq.Colormap('src_cmap'))