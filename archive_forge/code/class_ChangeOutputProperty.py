from Xlib import X
from Xlib.protocol import rq, structs
class ChangeOutputProperty(rq.Request):
    _request = rq.Struct(rq.Card8('opcode'), rq.Opcode(13), rq.RequestLength(), rq.Card32('output'), rq.Card32('property'), rq.Card32('type'), rq.Format('value', 1), rq.Card8('mode'), rq.Pad(2), rq.LengthOf('value', 4), rq.List('value', rq.Card8Obj))