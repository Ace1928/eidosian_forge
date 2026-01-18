from Xlib import X
from Xlib.protocol import rq, structs
class GetKeyboardMapping(rq.ReplyRequest):
    _request = rq.Struct(rq.Opcode(101), rq.Pad(1), rq.RequestLength(), rq.Card8('first_keycode'), rq.Card8('count'), rq.Pad(2))
    _reply = rq.Struct(rq.ReplyCode(), rq.Format('keysyms', 1), rq.Card16('sequence_number'), rq.ReplyLength(), rq.Pad(24), rq.KeyboardMapping('keysyms'))