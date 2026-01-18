from Xlib import X
from Xlib.protocol import rq, structs
class OutputChangeNotify(rq.Event):
    _code = None
    _fields = rq.Struct(rq.Card8('type'), rq.Card8('sub_code'), rq.Card16('sequence_number'), rq.Card32('timestamp'), rq.Card32('config_timestamp'), rq.Window('window'), rq.Card32('output'), rq.Card32('crtc'), rq.Card32('mode'), rq.Card16('rotation'), rq.Card8('connection'), rq.Card8('subpixel_order'))