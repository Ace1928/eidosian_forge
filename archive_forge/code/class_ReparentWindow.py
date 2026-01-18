from Xlib import X
from Xlib.protocol import rq, structs
class ReparentWindow(rq.Request):
    _request = rq.Struct(rq.Opcode(7), rq.Pad(1), rq.RequestLength(), rq.Window('window'), rq.Window('parent'), rq.Int16('x'), rq.Int16('y'))