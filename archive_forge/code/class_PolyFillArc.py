from Xlib import X
from Xlib.protocol import rq, structs
class PolyFillArc(rq.Request):
    _request = rq.Struct(rq.Opcode(71), rq.Pad(1), rq.RequestLength(), rq.Drawable('drawable'), rq.GC('gc'), rq.List('arcs', structs.Arc))