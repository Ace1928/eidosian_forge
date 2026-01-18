from Xlib import X
from Xlib.protocol import rq, structs
class ChangeSaveSet(rq.Request):
    _request = rq.Struct(rq.Opcode(6), rq.Set('mode', 1, (X.SetModeInsert, X.SetModeDelete)), rq.RequestLength(), rq.Window('window'))