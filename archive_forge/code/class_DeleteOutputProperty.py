from Xlib import X
from Xlib.protocol import rq, structs
class DeleteOutputProperty(rq.Request):
    _request = rq.Struct(rq.Card8('opcode'), rq.Opcode(14), rq.RequestLength(), rq.Card32('output'), rq.Card32('property'))