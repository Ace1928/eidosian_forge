from Xlib import X
from Xlib.protocol import rq, structs
class WarpPointer(rq.Request):
    _request = rq.Struct(rq.Opcode(41), rq.Pad(1), rq.RequestLength(), rq.Window('src_window'), rq.Window('dst_window'), rq.Int16('src_x'), rq.Int16('src_y'), rq.Card16('src_width'), rq.Card16('src_height'), rq.Int16('dst_x'), rq.Int16('dst_y'))