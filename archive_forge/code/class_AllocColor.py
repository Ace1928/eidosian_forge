from Xlib import X
from Xlib.protocol import rq, structs
class AllocColor(rq.ReplyRequest):
    _request = rq.Struct(rq.Opcode(84), rq.Pad(1), rq.RequestLength(), rq.Colormap('cmap'), rq.Card16('red'), rq.Card16('green'), rq.Card16('blue'), rq.Pad(2))
    _reply = rq.Struct(rq.ReplyCode(), rq.Pad(1), rq.Card16('sequence_number'), rq.ReplyLength(), rq.Card16('red'), rq.Card16('green'), rq.Card16('blue'), rq.Pad(2), rq.Card32('pixel'), rq.Pad(12))