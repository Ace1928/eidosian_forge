from Xlib import X
from Xlib.protocol import rq, structs
class CreateWindow(rq.Request):
    _request = rq.Struct(rq.Opcode(1), rq.Card8('depth'), rq.RequestLength(), rq.Window('wid'), rq.Window('parent'), rq.Int16('x'), rq.Int16('y'), rq.Card16('width'), rq.Card16('height'), rq.Card16('border_width'), rq.Set('window_class', 2, (X.CopyFromParent, X.InputOutput, X.InputOnly)), rq.Card32('visual'), structs.WindowValues('attrs'))