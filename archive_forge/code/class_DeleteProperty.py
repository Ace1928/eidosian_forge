from Xlib import X
from Xlib.protocol import rq, structs
class DeleteProperty(rq.Request):
    _request = rq.Struct(rq.Opcode(19), rq.Pad(1), rq.RequestLength(), rq.Window('window'), rq.Card32('property'))