from Xlib import X
from Xlib.protocol import rq, structs
class InstallColormap(rq.Request):
    _request = rq.Struct(rq.Opcode(81), rq.Pad(1), rq.RequestLength(), rq.Colormap('cmap'))