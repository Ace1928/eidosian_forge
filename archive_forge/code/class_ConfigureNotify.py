from Xlib import X
from Xlib.protocol import rq
class ConfigureNotify(rq.Event):
    _code = X.ConfigureNotify
    _fields = rq.Struct(rq.Card8('type'), rq.Pad(1), rq.Card16('sequence_number'), rq.Window('event'), rq.Window('window'), rq.Window('above_sibling', (X.NONE,)), rq.Int16('x'), rq.Int16('y'), rq.Card16('width'), rq.Card16('height'), rq.Card16('border_width'), rq.Card8('override'), rq.Pad(5))