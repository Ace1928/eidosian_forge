from Xlib import X
from Xlib.protocol import rq, structs
class FreeColormap(rq.Request):
    _request = rq.Struct(rq.Opcode(79), rq.Pad(1), rq.RequestLength(), rq.Colormap('cmap'))