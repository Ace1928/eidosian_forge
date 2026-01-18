from Xlib import X
from Xlib.protocol import rq, structs
class CreateGC(rq.Request):
    _request = rq.Struct(rq.Opcode(55), rq.Pad(1), rq.RequestLength(), rq.GC('cid'), rq.Drawable('drawable'), structs.GCValues('attrs'))