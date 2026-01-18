from Xlib import X
from Xlib.protocol import rq, structs
class CopyGC(rq.Request):
    _request = rq.Struct(rq.Opcode(57), rq.Pad(1), rq.RequestLength(), rq.GC('src_gc'), rq.GC('dst_gc'), rq.Card32('mask'))