from Xlib import X
from Xlib.protocol import rq, structs
class ListFonts(rq.ReplyRequest):
    _request = rq.Struct(rq.Opcode(49), rq.Pad(1), rq.RequestLength(), rq.Card16('max_names'), rq.LengthOf('pattern', 2), rq.String8('pattern'))
    _reply = rq.Struct(rq.ReplyCode(), rq.Pad(1), rq.Card16('sequence_number'), rq.ReplyLength(), rq.LengthOf('fonts', 2), rq.Pad(22), rq.List('fonts', rq.Str))