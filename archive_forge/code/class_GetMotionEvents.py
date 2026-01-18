from Xlib import X
from Xlib.protocol import rq, structs
class GetMotionEvents(rq.ReplyRequest):
    _request = rq.Struct(rq.Opcode(39), rq.Pad(1), rq.RequestLength(), rq.Window('window'), rq.Card32('start'), rq.Card32('stop'))
    _reply = rq.Struct(rq.ReplyCode(), rq.Pad(1), rq.Card16('sequence_number'), rq.ReplyLength(), rq.LengthOf('events', 4), rq.Pad(20), rq.List('events', structs.TimeCoord))