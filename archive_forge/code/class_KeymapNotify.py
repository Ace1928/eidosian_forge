from Xlib import X
from Xlib.protocol import rq
class KeymapNotify(rq.Event):
    _code = X.KeymapNotify
    _fields = rq.Struct(rq.Card8('type'), rq.FixedList('data', 31, rq.Card8Obj, pad=0))