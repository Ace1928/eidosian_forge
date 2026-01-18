from Xlib import X
from Xlib.protocol import rq
class KeyRelease(KeyButtonPointer):
    _code = X.KeyRelease