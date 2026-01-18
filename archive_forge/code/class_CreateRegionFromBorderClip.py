from Xlib import X
from Xlib.protocol import rq
from Xlib.xobject import drawable
class CreateRegionFromBorderClip(rq.Request):
    _request = rq.Struct(rq.Card8('opcode'), rq.Opcode(5), rq.RequestLength(), rq.Card32('region'), rq.Window('window'))