from Xlib import X
from Xlib.protocol import rq, structs
class DeleteOutputMode(rq.Request):
    _request = rq.Struct(rq.Card8('opcode'), rq.Opcode(19), rq.RequestLength(), rq.Card32('output'), rq.Card32('mode'))