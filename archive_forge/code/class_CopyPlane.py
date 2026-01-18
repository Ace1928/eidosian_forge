from Xlib import X
from Xlib.protocol import rq, structs
class CopyPlane(rq.Request):
    _request = rq.Struct(rq.Opcode(63), rq.Pad(1), rq.RequestLength(), rq.Drawable('src_drawable'), rq.Drawable('dst_drawable'), rq.GC('gc'), rq.Int16('src_x'), rq.Int16('src_y'), rq.Int16('dst_x'), rq.Int16('dst_y'), rq.Card16('width'), rq.Card16('height'), rq.Card32('bit_plane'))