from Xlib import X
from Xlib.protocol import rq, structs
class SetModifierMapping(rq.ReplyRequest):
    _request = rq.Struct(rq.Opcode(118), rq.Format('keycodes', 1), rq.RequestLength(), rq.ModifierMapping('keycodes'))
    _reply = rq.Struct(rq.ReplyCode(), rq.Card8('status'), rq.Card16('sequence_number'), rq.ReplyLength(), rq.Pad(24))