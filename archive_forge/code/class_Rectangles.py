from Xlib import X
from Xlib.protocol import rq, structs
class Rectangles(rq.Request):
    _request = rq.Struct(rq.Card8('opcode'), rq.Opcode(1), rq.RequestLength(), rq.Card8('operation'), rq.Set('region', 1, (ShapeBounding, ShapeClip)), rq.Card8('ordering'), rq.Pad(1), rq.Window('window'), rq.Int16('x'), rq.Int16('y'), rq.List('rectangles', structs.Rectangle))