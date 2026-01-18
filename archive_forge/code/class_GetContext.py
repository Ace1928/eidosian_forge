from Xlib import X
from Xlib.protocol import rq
class GetContext(rq.ReplyRequest):
    _request = rq.Struct(rq.Card8('opcode'), rq.Opcode(4), rq.RequestLength(), rq.Card32('context'))
    _reply = rq.Struct(rq.Pad(2), rq.Card16('sequence_number'), rq.ReplyLength(), rq.Card8('element_header'), rq.Pad(3), rq.LengthOf('client_info', 4), rq.Pad(16), rq.List('client_info', Record_ClientInfo))