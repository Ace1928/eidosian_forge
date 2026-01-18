from Xlib import X
from Xlib.protocol import rq
class RegisterClients(rq.Request):
    _request = rq.Struct(rq.Card8('opcode'), rq.Opcode(2), rq.RequestLength(), rq.Card32('context'), rq.Card8('element_header'), rq.Pad(3), rq.LengthOf('clients', 4), rq.LengthOf('ranges', 4), rq.List('clients', rq.Card32Obj), rq.List('ranges', Record_Range))