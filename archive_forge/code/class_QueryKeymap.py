from Xlib import X
from Xlib.protocol import rq, structs
class QueryKeymap(rq.ReplyRequest):
    _request = rq.Struct(rq.Opcode(44), rq.Pad(1), rq.RequestLength())
    _reply = rq.Struct(rq.ReplyCode(), rq.Pad(1), rq.Card16('sequence_number'), rq.ReplyLength(), rq.FixedList('map', 32, rq.Card8Obj))