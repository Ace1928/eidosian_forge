from Xlib import X
from Xlib.protocol import rq, structs
class GetState(rq.ReplyRequest):
    _request = rq.Struct(rq.Card8('opcode'), rq.Opcode(1), rq.RequestLength(), rq.Window('window'))
    _reply = rq.Struct(rq.ReplyCode(), rq.Bool('state'), rq.Card16('sequence_number'), rq.ReplyLength(), rq.Window('window'), rq.Pad(20))