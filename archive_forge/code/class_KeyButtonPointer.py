from Xlib import X
from Xlib.protocol import rq
class KeyButtonPointer(rq.Event):
    _code = None
    _fields = rq.Struct(rq.Card8('type'), rq.Card8('detail'), rq.Card16('sequence_number'), rq.Card32('time'), rq.Window('root'), rq.Window('window'), rq.Window('child', (X.NONE,)), rq.Int16('root_x'), rq.Int16('root_y'), rq.Int16('event_x'), rq.Int16('event_y'), rq.Card16('state'), rq.Card8('same_screen'), rq.Pad(1))