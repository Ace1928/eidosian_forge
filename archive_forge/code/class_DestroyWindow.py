from Xlib import X
from Xlib.protocol import rq, structs
class DestroyWindow(rq.Request):
    _request = rq.Struct(rq.Opcode(4), rq.Pad(1), rq.RequestLength(), rq.Window('window'))