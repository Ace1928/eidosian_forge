from Xlib import X
from Xlib.protocol import rq, structs
class ImageText16(rq.Request):
    _request = rq.Struct(rq.Opcode(77), rq.LengthOf('string', 1), rq.RequestLength(), rq.Drawable('drawable'), rq.GC('gc'), rq.Int16('x'), rq.Int16('y'), rq.String16('string'))