from Xlib import X
from Xlib.protocol import rq, structs
class SetScreenSaver(rq.Request):
    _request = rq.Struct(rq.Opcode(107), rq.Pad(1), rq.RequestLength(), rq.Int16('timeout'), rq.Int16('interval'), rq.Set('prefer_blank', 1, (X.DontPreferBlanking, X.PreferBlanking, X.DefaultBlanking)), rq.Set('allow_exposures', 1, (X.DontAllowExposures, X.AllowExposures, X.DefaultExposures)), rq.Pad(2))