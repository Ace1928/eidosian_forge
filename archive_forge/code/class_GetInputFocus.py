from Xlib import X
from Xlib.protocol import rq, structs
class GetInputFocus(rq.ReplyRequest):
    _request = rq.Struct(rq.Opcode(43), rq.Pad(1), rq.RequestLength())
    _reply = rq.Struct(rq.ReplyCode(), rq.Card8('revert_to'), rq.Card16('sequence_number'), rq.ReplyLength(), rq.Window('focus', (X.NONE, X.PointerRoot)), rq.Pad(20))