from Xlib import X
from Xlib.protocol import rq, structs
class InternAtom(rq.ReplyRequest):
    _request = rq.Struct(rq.Opcode(16), rq.Bool('only_if_exists'), rq.RequestLength(), rq.LengthOf('name', 2), rq.Pad(2), rq.String8('name'))
    _reply = rq.Struct(rq.ReplyCode(), rq.Pad(1), rq.Card16('sequence_number'), rq.ReplyLength(), rq.Card32('atom'), rq.Pad(20))