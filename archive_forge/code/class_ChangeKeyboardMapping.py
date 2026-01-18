from Xlib import X
from Xlib.protocol import rq, structs
class ChangeKeyboardMapping(rq.Request):
    _request = rq.Struct(rq.Opcode(100), rq.LengthOf('keysyms', 1), rq.RequestLength(), rq.Card8('first_keycode'), rq.Format('keysyms', 1), rq.Pad(2), rq.KeyboardMapping('keysyms'))