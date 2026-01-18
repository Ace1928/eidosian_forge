from Xlib import X
from Xlib.protocol import rq, structs
class CreatePixmap(rq.Request):
    _request = rq.Struct(rq.Opcode(53), rq.Card8('depth'), rq.RequestLength(), rq.Pixmap('pid'), rq.Drawable('drawable'), rq.Card16('width'), rq.Card16('height'))