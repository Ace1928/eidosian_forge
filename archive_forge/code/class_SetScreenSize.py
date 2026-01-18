from Xlib import X
from Xlib.protocol import rq, structs
class SetScreenSize(rq.Request):
    _request = rq.Struct(rq.Card8('opcode'), rq.Opcode(7), rq.RequestLength(), rq.Window('window'), rq.Card16('width'), rq.Card16('height'), rq.Card32('width_in_millimeters'), rq.Card32('height_in_millimeters'))