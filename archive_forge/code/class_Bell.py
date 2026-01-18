from Xlib import X
from Xlib.protocol import rq, structs
class Bell(rq.Request):
    _request = rq.Struct(rq.Opcode(104), rq.Int8('percent'), rq.RequestLength())