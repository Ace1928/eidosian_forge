from Xlib import X
from Xlib.protocol import rq, structs
class Offset(rq.Request):
    _request = rq.Struct(rq.Card8('opcode'), rq.Opcode(4), rq.RequestLength(), rq.Set('region', 1, (ShapeBounding, ShapeClip)), rq.Pad(3), rq.Window('window'), rq.Int16('x'), rq.Int16('y'))