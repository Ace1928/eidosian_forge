from Xlib import X
from Xlib.protocol import rq, structs
class QueryTextExtents(rq.ReplyRequest):
    _request = rq.Struct(rq.Opcode(48), rq.OddLength('string'), rq.RequestLength(), rq.Fontable('font'), rq.String16('string'))
    _reply = rq.Struct(rq.ReplyCode(), rq.Card8('draw_direction'), rq.Card16('sequence_number'), rq.ReplyLength(), rq.Int16('font_ascent'), rq.Int16('font_descent'), rq.Int16('overall_ascent'), rq.Int16('overall_descent'), rq.Int32('overall_width'), rq.Int32('overall_left'), rq.Int32('overall_right'), rq.Pad(4))