from Xlib import X
from Xlib.protocol import rq, structs
class InputSelected(rq.ReplyRequest):
    _request = rq.Struct(rq.Card8('opcode'), rq.Opcode(7), rq.RequestLength(), rq.Window('window'))
    _reply = rq.Struct(rq.ReplyCode(), rq.Bool('enabled'), rq.Card16('sequence_number'), rq.ReplyLength(), rq.Pad(24))