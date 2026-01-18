from Xlib import X
from Xlib.protocol import rq, structs
class FreeColors(rq.Request):
    _request = rq.Struct(rq.Opcode(88), rq.Pad(1), rq.RequestLength(), rq.Colormap('cmap'), rq.Card32('plane_mask'), rq.List('pixels', rq.Card32Obj))