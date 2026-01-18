from Xlib import X
from Xlib.protocol import rq
from Xlib.xobject import drawable
class RedirectWindow(rq.Request):
    _request = rq.Struct(rq.Card8('opcode'), rq.Opcode(1), rq.RequestLength(), rq.Window('window'), rq.Set('update', 1, (RedirectAutomatic, RedirectManual)), rq.Pad(3))