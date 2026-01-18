from Xlib import X
from Xlib.protocol import rq, structs
class ChangeActivePointerGrab(rq.Request):
    _request = rq.Struct(rq.Opcode(30), rq.Pad(1), rq.RequestLength(), rq.Cursor('cursor'), rq.Card32('time'), rq.Card16('event_mask'), rq.Pad(2))