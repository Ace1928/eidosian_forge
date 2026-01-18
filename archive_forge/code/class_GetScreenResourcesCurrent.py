from Xlib import X
from Xlib.protocol import rq, structs
class GetScreenResourcesCurrent(rq.ReplyRequest):
    _request = rq.Struct(rq.Card8('opcode'), rq.Opcode(25), rq.RequestLength(), rq.Window('window'))
    _reply = rq.Struct(rq.ReplyCode(), rq.Pad(1), rq.Card16('sequence_number'), rq.ReplyLength(), rq.Card32('timestamp'), rq.Card32('config_timestamp'), rq.LengthOf('crtcs', 2), rq.LengthOf('outputs', 2), rq.LengthOf('modes', 2), rq.LengthOf('names', 2), rq.Pad(8), rq.List('crtcs', rq.Card32Obj), rq.List('outputs', rq.Card32Obj), rq.List('modes', RandR_ModeInfo), rq.String8('names'))