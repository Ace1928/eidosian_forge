from Xlib import X
from Xlib.protocol import rq, structs
class QueryColors(rq.ReplyRequest):
    _request = rq.Struct(rq.Opcode(91), rq.Pad(1), rq.RequestLength(), rq.Colormap('cmap'), rq.List('pixels', rq.Card32Obj))
    _reply = rq.Struct(rq.ReplyCode(), rq.Pad(1), rq.Card16('sequence_number'), rq.ReplyLength(), rq.LengthOf('colors', 2), rq.Pad(22), rq.List('colors', structs.RGB))