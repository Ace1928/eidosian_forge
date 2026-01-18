from Xlib import X
from Xlib.protocol import rq
class EnterNotify(EnterLeave):
    _code = X.EnterNotify