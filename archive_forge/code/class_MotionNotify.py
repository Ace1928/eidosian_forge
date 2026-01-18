from Xlib import X
from Xlib.protocol import rq
class MotionNotify(KeyButtonPointer):
    _code = X.MotionNotify