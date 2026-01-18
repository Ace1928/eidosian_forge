from Xlib import X
from Xlib.protocol import rq, structs
class GetPointerMapping(rq.ReplyRequest):
    _request = rq.Struct(rq.Opcode(117), rq.Pad(1), rq.RequestLength())
    _reply = rq.Struct(rq.ReplyCode(), rq.LengthOf('map', 1), rq.Card16('sequence_number'), rq.ReplyLength(), rq.Pad(24), rq.List('map', rq.Card8Obj))