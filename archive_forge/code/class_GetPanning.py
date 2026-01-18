from Xlib import X
from Xlib.protocol import rq, structs
class GetPanning(rq.ReplyRequest):
    _request = rq.Struct(rq.Card8('opcode'), rq.Opcode(28), rq.RequestLength(), rq.Card32('crtc'))
    _reply = rq.Struct(rq.ReplyCode(), rq.Card8('status'), rq.Card16('sequence_number'), rq.ReplyLength(), rq.Card32('timestamp'), rq.Card16('left'), rq.Card16('top'), rq.Card16('width'), rq.Card16('height'), rq.Card16('track_left'), rq.Card16('track_top'), rq.Card16('track_width'), rq.Card16('track_height'), rq.Int16('border_left'), rq.Int16('border_top'), rq.Int16('border_right'), rq.Int16('border_bottom'))