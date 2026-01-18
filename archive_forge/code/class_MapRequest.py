from Xlib import X
from Xlib.protocol import rq
class MapRequest(rq.Event):
    _code = X.MapRequest
    _fields = rq.Struct(rq.Card8('type'), rq.Pad(1), rq.Card16('sequence_number'), rq.Window('parent'), rq.Window('window'), rq.Pad(20))