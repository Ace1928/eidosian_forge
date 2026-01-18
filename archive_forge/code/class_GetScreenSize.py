from Xlib import X
from Xlib.protocol import rq, structs
class GetScreenSize(rq.ReplyRequest):
    _request = rq.Struct(rq.Card8('opcode'), rq.Opcode(3), rq.RequestLength(), rq.Window('window'), rq.Card32('screen'))
    _reply = rq.Struct(rq.ReplyCode(), rq.Pad(1), rq.Card16('sequence_number'), rq.Card32('length'), rq.Card32('width'), rq.Card32('height'), rq.Window('window'), rq.Card32('screen'), rq.Pad(8))