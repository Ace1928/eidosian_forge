from Xlib import X
from Xlib.protocol import rq
class MappingNotify(rq.Event):
    _code = X.MappingNotify
    _fields = rq.Struct(rq.Card8('type'), rq.Pad(1), rq.Card16('sequence_number'), rq.Card8('request'), rq.Card8('first_keycode'), rq.Card8('count'), rq.Pad(25))