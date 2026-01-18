from Xlib import X
from Xlib.protocol import rq
class CompareCursor(rq.ReplyRequest):
    _request = rq.Struct(rq.Card8('opcode'), rq.Opcode(1), rq.RequestLength(), rq.Window('window'), rq.Cursor('cursor', (X.NONE, CurrentCursor)))
    _reply = rq.Struct(rq.Pad(1), rq.Card8('same'), rq.Card16('sequence_number'), rq.Pad(28))