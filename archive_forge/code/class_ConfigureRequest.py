from Xlib import X
from Xlib.protocol import rq
class ConfigureRequest(rq.Event):
    _code = X.ConfigureRequest
    _fields = rq.Struct(rq.Card8('type'), rq.Card8('stack_mode'), rq.Card16('sequence_number'), rq.Window('parent'), rq.Window('window'), rq.Window('sibling', (X.NONE,)), rq.Int16('x'), rq.Int16('y'), rq.Card16('width'), rq.Card16('height'), rq.Card16('border_width'), rq.Card16('value_mask'), rq.Pad(4))