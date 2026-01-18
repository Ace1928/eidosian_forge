from Xlib import X
from Xlib.protocol import rq, structs
class OpenFont(rq.Request):
    _request = rq.Struct(rq.Opcode(45), rq.Pad(1), rq.RequestLength(), rq.Font('fid'), rq.LengthOf('name', 2), rq.Pad(2), rq.String8('name'))