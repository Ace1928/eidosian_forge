from Xlib import X
from Xlib.protocol import rq
class UnmapNotify(rq.Event):
    _code = X.UnmapNotify
    _fields = rq.Struct(rq.Card8('type'), rq.Pad(1), rq.Card16('sequence_number'), rq.Window('event'), rq.Window('window'), rq.Card8('from_configure'), rq.Pad(19))