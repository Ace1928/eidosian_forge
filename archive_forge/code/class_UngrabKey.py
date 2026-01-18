from Xlib import X
from Xlib.protocol import rq, structs
class UngrabKey(rq.Request):
    _request = rq.Struct(rq.Opcode(34), rq.Card8('key'), rq.RequestLength(), rq.Window('grab_window'), rq.Card16('modifiers'), rq.Pad(2))