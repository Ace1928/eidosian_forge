from Xlib import X
from Xlib.protocol import rq, structs
class CrtcChangeNotify(rq.Event):
    _code = None
    _fields = rq.Struct(rq.Card8('type'), rq.Card8('sub_code'), rq.Card16('sequence_number'), rq.Card32('timestamp'), rq.Window('window'), rq.Card32('crtc'), rq.Card32('mode'), rq.Card16('rotation'), rq.Pad(2), rq.Int16('x'), rq.Int16('y'), rq.Card16('width'), rq.Card16('height'))