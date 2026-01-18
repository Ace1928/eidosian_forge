from Xlib import X
from Xlib.protocol import rq, structs
class UngrabServer(rq.Request):
    _request = rq.Struct(rq.Opcode(37), rq.Pad(1), rq.RequestLength())