from Xlib import X
from Xlib.protocol import rq
class GraphicsExpose(rq.Event):
    _code = X.GraphicsExpose
    _fields = rq.Struct(rq.Card8('type'), rq.Pad(1), rq.Card16('sequence_number'), rq.Drawable('drawable'), rq.Card16('x'), rq.Card16('y'), rq.Card16('width'), rq.Card16('height'), rq.Card16('minor_event'), rq.Card16('count'), rq.Card8('major_event'), rq.Pad(11))