from Xlib import X
from Xlib.protocol import rq
class SelectionClear(rq.Event):
    _code = X.SelectionClear
    _fields = rq.Struct(rq.Card8('type'), rq.Pad(1), rq.Card16('sequence_number'), rq.Card32('time'), rq.Window('window'), rq.Card32('atom'), rq.Pad(16))