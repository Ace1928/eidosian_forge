from Xlib import X
from Xlib.protocol import rq, structs
class ChangeProperty(rq.Request):
    _request = rq.Struct(rq.Opcode(18), rq.Set('mode', 1, (X.PropModeReplace, X.PropModePrepend, X.PropModeAppend)), rq.RequestLength(), rq.Window('window'), rq.Card32('property'), rq.Card32('type'), rq.Format('data', 1), rq.Pad(3), rq.LengthOf('data', 4), rq.PropertyData('data'))