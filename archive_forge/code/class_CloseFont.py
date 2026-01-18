from Xlib import X
from Xlib.protocol import rq, structs
class CloseFont(rq.Request):
    _request = rq.Struct(rq.Opcode(46), rq.Pad(1), rq.RequestLength(), rq.Font('font'))