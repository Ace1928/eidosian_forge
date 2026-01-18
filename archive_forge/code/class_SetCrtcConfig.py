from Xlib import X
from Xlib.protocol import rq, structs
class SetCrtcConfig(rq.ReplyRequest):
    _request = rq.Struct(rq.Card8('opcode'), rq.Opcode(21), rq.RequestLength(), rq.Card32('crtc'), rq.Card32('timestamp'), rq.Card32('config_timestamp'), rq.Int16('x'), rq.Int16('y'), rq.Card32('mode'), rq.Card16('rotation'), rq.Pad(2), rq.List('outputs', rq.Card32Obj))
    _reply = rq.Struct(rq.ReplyCode(), rq.Card8('status'), rq.Card16('sequence_number'), rq.ReplyLength(), rq.Card32('new_timestamp'), rq.Pad(20))