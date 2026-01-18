from Xlib import X
from Xlib.protocol import rq, structs
class GetKeyboardControl(rq.ReplyRequest):
    _request = rq.Struct(rq.Opcode(103), rq.Pad(1), rq.RequestLength())
    _reply = rq.Struct(rq.ReplyCode(), rq.Card8('global_auto_repeat'), rq.Card16('sequence_number'), rq.ReplyLength(), rq.Card32('led_mask'), rq.Card8('key_click_percent'), rq.Card8('bell_percent'), rq.Card16('bell_pitch'), rq.Card16('bell_duration'), rq.Pad(2), rq.FixedList('auto_repeats', 32, rq.Card8Obj))