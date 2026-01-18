from Xlib import X
from Xlib.protocol import rq, structs
class GetFontPath(rq.ReplyRequest):
    _request = rq.Struct(rq.Opcode(52), rq.Pad(1), rq.RequestLength())
    _reply = rq.Struct(rq.ReplyCode(), rq.Pad(1), rq.Card16('sequence_number'), rq.ReplyLength(), rq.LengthOf('paths', 2), rq.Pad(22), rq.List('paths', rq.Str))