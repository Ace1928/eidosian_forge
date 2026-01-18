from Xlib import X
from Xlib.protocol import rq, structs
class QueryOutputProperty(rq.ReplyRequest):
    _request = rq.Struct(rq.Card8('opcode'), rq.Opcode(11), rq.RequestLength(), rq.Card32('output'), rq.Card32('property'))
    _reply = rq.Struct(rq.ReplyCode(), rq.Pad(1), rq.Card16('sequence_number'), rq.ReplyLength(), rq.Bool('pending'), rq.Bool('range'), rq.Bool('immutable'), rq.Pad(21), rq.List('valid_values', rq.Card32Obj))