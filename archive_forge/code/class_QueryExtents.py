from Xlib import X
from Xlib.protocol import rq, structs
class QueryExtents(rq.ReplyRequest):
    _request = rq.Struct(rq.Card8('opcode'), rq.Opcode(5), rq.RequestLength(), rq.Window('window'))
    _reply = rq.Struct(rq.ReplyCode(), rq.Pad(1), rq.Card16('sequence_number'), rq.ReplyLength(), rq.Bool('bounding_shaped'), rq.Bool('clip_shaped'), rq.Pad(2), rq.Int16('bounding_x'), rq.Int16('bounding_y'), rq.Card16('bounding_width'), rq.Card16('bounding_height'), rq.Int16('clip_x'), rq.Int16('clip_y'), rq.Card16('clip_width'), rq.Card16('clip_height'), rq.Pad(4))