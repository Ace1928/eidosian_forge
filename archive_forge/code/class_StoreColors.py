from Xlib import X
from Xlib.protocol import rq, structs
class StoreColors(rq.Request):
    _request = rq.Struct(rq.Opcode(89), rq.Pad(1), rq.RequestLength(), rq.Colormap('cmap'), rq.List('items', structs.ColorItem))