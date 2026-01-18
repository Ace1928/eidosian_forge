from Xlib import X
from Xlib.protocol import rq
class VisibilityNotify(rq.Event):
    _code = X.VisibilityNotify
    _fields = rq.Struct(rq.Card8('type'), rq.Pad(1), rq.Card16('sequence_number'), rq.Window('window'), rq.Card8('state'), rq.Pad(23))