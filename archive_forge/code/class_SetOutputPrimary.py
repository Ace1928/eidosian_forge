from Xlib import X
from Xlib.protocol import rq, structs
class SetOutputPrimary(rq.Request):
    _request = rq.Struct(rq.Card8('opcode'), rq.Opcode(30), rq.RequestLength(), rq.Window('window'), rq.Card32('output'))