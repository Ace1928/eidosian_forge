from Xlib import X
from Xlib.protocol import rq, structs
class MapWindow(rq.Request):
    _request = rq.Struct(rq.Opcode(8), rq.Pad(1), rq.RequestLength(), rq.Window('window'))