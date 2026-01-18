from Xlib import X
from Xlib.protocol import rq, structs
class QueryTree(rq.ReplyRequest):
    _request = rq.Struct(rq.Opcode(15), rq.Pad(1), rq.RequestLength(), rq.Window('window'))
    _reply = rq.Struct(rq.ReplyCode(), rq.Pad(1), rq.Card16('sequence_number'), rq.ReplyLength(), rq.Window('root'), rq.Window('parent', (X.NONE,)), rq.LengthOf('children', 2), rq.Pad(14), rq.List('children', rq.WindowObj))