from Xlib import X
from Xlib.protocol import rq
class DestroyNotify(rq.Event):
    _code = X.DestroyNotify
    _fields = rq.Struct(rq.Card8('type'), rq.Pad(1), rq.Card16('sequence_number'), rq.Window('event'), rq.Window('window'), rq.Pad(20))