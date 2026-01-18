from Xlib import X
from Xlib.protocol import rq, structs
class CreateCursor(rq.Request):
    _request = rq.Struct(rq.Opcode(93), rq.Pad(1), rq.RequestLength(), rq.Cursor('cid'), rq.Pixmap('source'), rq.Pixmap('mask'), rq.Card16('fore_red'), rq.Card16('fore_green'), rq.Card16('fore_blue'), rq.Card16('back_red'), rq.Card16('back_green'), rq.Card16('back_blue'), rq.Card16('x'), rq.Card16('y'))