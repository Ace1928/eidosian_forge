from Xlib import X
from Xlib.protocol import rq, structs
class AllocColorCells(rq.ReplyRequest):
    _request = rq.Struct(rq.Opcode(86), rq.Bool('contiguous'), rq.RequestLength(), rq.Colormap('cmap'), rq.Card16('colors'), rq.Card16('planes'))
    _reply = rq.Struct(rq.ReplyCode(), rq.Pad(1), rq.Card16('sequence_number'), rq.ReplyLength(), rq.LengthOf('pixels', 2), rq.LengthOf('masks', 2), rq.Pad(20), rq.List('pixels', rq.Card32Obj), rq.List('masks', rq.Card32Obj))