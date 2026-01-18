from Xlib import X
from Xlib.protocol import rq, structs
class FillPoly(rq.Request):
    _request = rq.Struct(rq.Opcode(69), rq.Pad(1), rq.RequestLength(), rq.Drawable('drawable'), rq.GC('gc'), rq.Set('shape', 1, (X.Complex, X.Nonconvex, X.Convex)), rq.Set('coord_mode', 1, (X.CoordModeOrigin, X.CoordModePrevious)), rq.Pad(2), rq.List('points', structs.Point))