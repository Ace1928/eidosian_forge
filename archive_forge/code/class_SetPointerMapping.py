from Xlib import X
from Xlib.protocol import rq, structs
class SetPointerMapping(rq.ReplyRequest):
    _request = rq.Struct(rq.Opcode(116), rq.LengthOf('map', 1), rq.RequestLength(), rq.List('map', rq.Card8Obj))
    _reply = rq.Struct(rq.ReplyCode(), rq.Card8('status'), rq.Card16('sequence_number'), rq.ReplyLength(), rq.Pad(24))