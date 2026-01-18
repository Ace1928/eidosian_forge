from Xlib import X
from Xlib.protocol import rq, structs
class UnmapWindow(rq.Request):
    _request = rq.Struct(rq.Opcode(10), rq.Pad(1), rq.RequestLength(), rq.Window('window'))