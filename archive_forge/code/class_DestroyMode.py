from Xlib import X
from Xlib.protocol import rq, structs
class DestroyMode(rq.Request):
    _request = rq.Struct(rq.Card8('opcode'), rq.Opcode(17), rq.RequestLength(), rq.Card32('mode'))