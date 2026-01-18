from Xlib import X
from Xlib.protocol import rq, structs
class GetOutputPrimary(rq.ReplyRequest):
    _request = rq.Struct(rq.Card8('opcode'), rq.Opcode(31), rq.RequestLength(), rq.Window('window'))
    _reply = rq.Struct(rq.ReplyCode(), rq.Pad(1), rq.Card16('sequence_number'), rq.ReplyLength(), rq.Card32('output'), rq.Pad(20))