from Xlib import X
from Xlib.protocol import rq, structs
class PolySegment(rq.Request):
    _request = rq.Struct(rq.Opcode(66), rq.Pad(1), rq.RequestLength(), rq.Drawable('drawable'), rq.GC('gc'), rq.List('segments', structs.Segment))