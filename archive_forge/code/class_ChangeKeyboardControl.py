from Xlib import X
from Xlib.protocol import rq, structs
class ChangeKeyboardControl(rq.Request):
    _request = rq.Struct(rq.Opcode(102), rq.Pad(1), rq.RequestLength(), rq.ValueList('attrs', 4, 0, rq.Int8('key_click_percent'), rq.Int8('bell_percent'), rq.Int16('bell_pitch'), rq.Int16('bell_duration'), rq.Card8('led'), rq.Set('led_mode', 1, (X.LedModeOff, X.LedModeOn)), rq.Card8('key'), rq.Set('auto_repeat_mode', 1, (X.AutoRepeatModeOff, X.AutoRepeatModeOn, X.AutoRepeatModeDefault))))