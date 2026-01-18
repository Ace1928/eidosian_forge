from Xlib import X
from Xlib.protocol import rq, structs
class PolyRectangle(rq.Request):
    _request = rq.Struct(rq.Opcode(67), rq.Pad(1), rq.RequestLength(), rq.Drawable('drawable'), rq.GC('gc'), rq.List('rectangles', structs.Rectangle))