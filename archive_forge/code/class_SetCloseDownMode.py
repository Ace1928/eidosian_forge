from Xlib import X
from Xlib.protocol import rq, structs
class SetCloseDownMode(rq.Request):
    _request = rq.Struct(rq.Opcode(112), rq.Set('mode', 1, (X.DestroyAll, X.RetainPermanent, X.RetainTemporary)), rq.RequestLength())