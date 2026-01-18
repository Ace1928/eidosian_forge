from Xlib import X
from Xlib.protocol import rq
class NoExpose(rq.Event):
    _code = X.NoExpose
    _fields = rq.Struct(rq.Card8('type'), rq.Pad(1), rq.Card16('sequence_number'), rq.Drawable('window'), rq.Card16('minor_event'), rq.Card8('major_event'), rq.Pad(21))