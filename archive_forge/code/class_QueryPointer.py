from Xlib import X
from Xlib.protocol import rq, structs
class QueryPointer(rq.ReplyRequest):
    _request = rq.Struct(rq.Opcode(38), rq.Pad(1), rq.RequestLength(), rq.Window('window'))
    _reply = rq.Struct(rq.ReplyCode(), rq.Card8('same_screen'), rq.Card16('sequence_number'), rq.ReplyLength(), rq.Window('root'), rq.Window('child', (X.NONE,)), rq.Int16('root_x'), rq.Int16('root_y'), rq.Int16('win_x'), rq.Int16('win_y'), rq.Card16('mask'), rq.Pad(6))