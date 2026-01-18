from Xlib import X
from Xlib.protocol import rq, structs
class DestroySubWindows(rq.Request):
    _request = rq.Struct(rq.Opcode(5), rq.Pad(1), rq.RequestLength(), rq.Window('window'))