from Xlib import X
from Xlib.protocol import rq
class GravityNotify(rq.Event):
    _code = X.GravityNotify
    _fields = rq.Struct(rq.Card8('type'), rq.Pad(1), rq.Card16('sequence_number'), rq.Window('event'), rq.Window('window'), rq.Int16('x'), rq.Int16('y'), rq.Pad(16))