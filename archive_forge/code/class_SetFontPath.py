from Xlib import X
from Xlib.protocol import rq, structs
class SetFontPath(rq.Request):
    _request = rq.Struct(rq.Opcode(51), rq.Pad(1), rq.RequestLength(), rq.LengthOf('path', 2), rq.Pad(2), rq.List('path', rq.Str))