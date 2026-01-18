from Xlib import X
from Xlib.protocol import rq, structs
class UninstallColormap(rq.Request):
    _request = rq.Struct(rq.Opcode(82), rq.Pad(1), rq.RequestLength(), rq.Colormap('cmap'))