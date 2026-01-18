from Xlib import X
from Xlib.protocol import rq, structs
class GetModifierMapping(rq.ReplyRequest):
    _request = rq.Struct(rq.Opcode(119), rq.Pad(1), rq.RequestLength())
    _reply = rq.Struct(rq.ReplyCode(), rq.Format('keycodes', 1), rq.Card16('sequence_number'), rq.ReplyLength(), rq.Pad(24), rq.ModifierMapping('keycodes'))