from Xlib import X
from Xlib.protocol import rq, structs
class GetImage(rq.ReplyRequest):
    _request = rq.Struct(rq.Opcode(73), rq.Set('format', 1, (X.XYPixmap, X.ZPixmap)), rq.RequestLength(), rq.Drawable('drawable'), rq.Int16('x'), rq.Int16('y'), rq.Card16('width'), rq.Card16('height'), rq.Card32('plane_mask'))
    _reply = rq.Struct(rq.ReplyCode(), rq.Card8('depth'), rq.Card16('sequence_number'), rq.ReplyLength(), rq.Card32('visual'), rq.Pad(20), rq.String8('data'))