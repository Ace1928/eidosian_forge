from Xlib import X
from Xlib.protocol import rq, structs
class GetCrtcGammaSize(rq.ReplyRequest):
    _request = rq.Struct(rq.Card8('opcode'), rq.Opcode(22), rq.RequestLength(), rq.Card32('crtc'))
    _reply = rq.Struct(rq.ReplyCode(), rq.Card8('status'), rq.Card16('sequence_number'), rq.ReplyLength(), rq.Card16('size'), rq.Pad(22))