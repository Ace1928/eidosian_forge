from Xlib import X
from Xlib.protocol import rq, structs
class ConfigureOutputProperty(rq.Request):
    _request = rq.Struct(rq.Card8('opcode'), rq.Opcode(12), rq.RequestLength(), rq.Card32('output'), rq.Card32('property'), rq.Bool('pending'), rq.Bool('range'), rq.Pad(2), rq.List('valid_values', rq.Card32Obj))