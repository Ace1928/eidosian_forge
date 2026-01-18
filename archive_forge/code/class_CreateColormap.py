from Xlib import X
from Xlib.protocol import rq, structs
class CreateColormap(rq.Request):
    _request = rq.Struct(rq.Opcode(78), rq.Set('alloc', 1, (X.AllocNone, X.AllocAll)), rq.RequestLength(), rq.Colormap('mid'), rq.Window('window'), rq.Card32('visual'))