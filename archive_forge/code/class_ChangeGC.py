from Xlib import X
from Xlib.protocol import rq, structs
class ChangeGC(rq.Request):
    _request = rq.Struct(rq.Opcode(56), rq.Pad(1), rq.RequestLength(), rq.GC('gc'), structs.GCValues('attrs'))