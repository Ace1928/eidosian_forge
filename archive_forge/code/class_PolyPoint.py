from Xlib import X
from Xlib.protocol import rq, structs
class PolyPoint(rq.Request):
    _request = rq.Struct(rq.Opcode(64), rq.Set('coord_mode', 1, (X.CoordModeOrigin, X.CoordModePrevious)), rq.RequestLength(), rq.Drawable('drawable'), rq.GC('gc'), rq.List('points', structs.Point))