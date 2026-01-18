from Xlib import X
from Xlib.protocol import rq, structs
class StoreNamedColor(rq.Request):
    _request = rq.Struct(rq.Opcode(90), rq.Card8('flags'), rq.RequestLength(), rq.Colormap('cmap'), rq.Card32('pixel'), rq.LengthOf('name', 2), rq.Pad(2), rq.String8('name'))