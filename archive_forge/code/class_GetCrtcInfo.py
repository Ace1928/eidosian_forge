from Xlib import X
from Xlib.protocol import rq, structs
class GetCrtcInfo(rq.ReplyRequest):
    _request = rq.Struct(rq.Card8('opcode'), rq.Opcode(20), rq.RequestLength(), rq.Card32('crtc'), rq.Card32('config_timestamp'))
    _reply = rq.Struct(rq.ReplyCode(), rq.Card8('status'), rq.Card16('sequence_number'), rq.ReplyLength(), rq.Card32('timestamp'), rq.Card16('width'), rq.Card16('height'), rq.Card32('mode'), rq.Card16('rotation'), rq.Card16('possible_rotations'), rq.LengthOf('outputs', 2), rq.LengthOf('possible_outputs', 2), rq.List('outputs', rq.Card32Obj), rq.List('possible_outputs', rq.Card32Obj))