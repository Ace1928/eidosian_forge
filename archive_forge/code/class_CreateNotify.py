from Xlib import X
from Xlib.protocol import rq
class CreateNotify(rq.Event):
    _code = X.CreateNotify
    _fields = rq.Struct(rq.Card8('type'), rq.Pad(1), rq.Card16('sequence_number'), rq.Window('parent'), rq.Window('window'), rq.Int16('x'), rq.Int16('y'), rq.Card16('width'), rq.Card16('height'), rq.Card16('border_width'), rq.Card8('override'), rq.Pad(9))