from Xlib import X
from Xlib.protocol import rq
class XResourceError(XError):
    _fields = rq.Struct(rq.Card8('type'), rq.Card8('code'), rq.Card16('sequence_number'), rq.Resource('resource_id'), rq.Card16('minor_opcode'), rq.Card8('major_opcode'), rq.Pad(21))