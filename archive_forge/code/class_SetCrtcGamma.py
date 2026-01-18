from Xlib import X
from Xlib.protocol import rq, structs
class SetCrtcGamma(rq.Request):
    _request = rq.Struct(rq.Card8('opcode'), rq.Opcode(24), rq.RequestLength(), rq.Card32('crtc'), rq.Card16('size'), rq.Pad(2), rq.List('red', rq.Card16Obj), rq.List('green', rq.Card16Obj), rq.List('blue', rq.Card16Obj))